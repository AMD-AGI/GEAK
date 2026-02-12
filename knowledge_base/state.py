"""
State Machine for mini-kernel Agent

States:
- AUTONOMOUS: Running optimization autonomously
- CHECKPOINT: Saving a checkpoint after success
- INTERVENE: Paused, asking user for input
- ERROR: Handling an error
- COMPLETE: Optimization finished

Transitions:
- AUTONOMOUS → CHECKPOINT: After successful optimization step
- AUTONOMOUS → INTERVENE: On trigger (failure, regression, etc.)
- AUTONOMOUS → ERROR: On critical error
- CHECKPOINT → AUTONOMOUS: Resume after save
- INTERVENE → AUTONOMOUS: User says continue
- ERROR → AUTONOMOUS: After recovery
- Any → COMPLETE: Optimization finished
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
import time


class AgentState(Enum):
    """Agent states in the optimization pipeline."""
    INIT = auto()          # Initial state
    DISCOVERING = auto()   # Discovering tests and baseline
    AUTONOMOUS = auto()    # Running autonomously
    CHECKPOINT = auto()    # Saving checkpoint
    INTERVENE = auto()     # Waiting for user input
    ERROR = auto()         # Handling error
    COMPLETE = auto()      # Finished


class TransitionTrigger(Enum):
    """Triggers for state transitions."""
    # Success triggers
    TESTS_PASS = auto()
    PERFORMANCE_IMPROVED = auto()
    CHECKPOINT_SAVED = auto()
    
    # Intervention triggers
    TESTS_FAIL_REPEATED = auto()
    PERFORMANCE_REGRESSION = auto()
    MAJOR_CHANGE = auto()
    USER_INTERRUPT = auto()
    TIME_WITHOUT_PROGRESS = auto()
    AGENT_UNCERTAIN = auto()
    COST_LIMIT = auto()
    
    # Error triggers
    CRITICAL_ERROR = auto()
    PROTECTED_FILE_TOUCHED = auto()
    
    # Completion triggers
    GOAL_REACHED = auto()
    MAX_ITERATIONS = auto()
    USER_QUIT = auto()


@dataclass
class StateTransition:
    """A state transition record."""
    from_state: AgentState
    to_state: AgentState
    trigger: TransitionTrigger
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationProgress:
    """Current optimization progress."""
    iteration: int = 0
    baseline_latency_us: float = 0.0
    current_latency_us: float = 0.0
    best_latency_us: float = float('inf')
    best_checkpoint: Optional[str] = None
    speedup_vs_baseline: float = 1.0
    speedup_vs_best: float = 1.0
    consecutive_failures: int = 0
    time_since_improvement: float = 0.0
    total_cost: float = 0.0
    strategies_tried: List[str] = field(default_factory=list)


class StateMachine:
    """
    State machine for mini-kernel agent.
    
    Manages transitions between states based on triggers
    and intervention policy.
    """
    
    # Valid state transitions
    VALID_TRANSITIONS = {
        AgentState.INIT: [AgentState.DISCOVERING, AgentState.ERROR],
        AgentState.DISCOVERING: [AgentState.AUTONOMOUS, AgentState.INTERVENE, AgentState.ERROR],
        AgentState.AUTONOMOUS: [AgentState.CHECKPOINT, AgentState.INTERVENE, AgentState.ERROR, AgentState.COMPLETE],
        AgentState.CHECKPOINT: [AgentState.AUTONOMOUS, AgentState.INTERVENE, AgentState.COMPLETE],
        AgentState.INTERVENE: [AgentState.AUTONOMOUS, AgentState.COMPLETE],
        AgentState.ERROR: [AgentState.AUTONOMOUS, AgentState.INTERVENE, AgentState.COMPLETE],
        AgentState.COMPLETE: [],  # Terminal state
    }
    
    def __init__(self, intervention_policy: 'InterventionPolicy' = None):
        self.state = AgentState.INIT
        self.history: List[StateTransition] = []
        self.progress = OptimizationProgress()
        self.policy = intervention_policy
        self._last_improvement_time = time.time()
        self._callbacks: Dict[AgentState, List[Callable]] = {}
    
    def transition(self, trigger: TransitionTrigger, 
                  details: Dict[str, Any] = None) -> AgentState:
        """
        Attempt a state transition based on trigger.
        
        Returns the new state (may be same as current if transition invalid).
        """
        details = details or {}
        new_state = self._determine_new_state(trigger, details)
        
        if new_state and new_state in self.VALID_TRANSITIONS.get(self.state, []):
            old_state = self.state
            self.state = new_state
            
            # Record transition
            transition = StateTransition(
                from_state=old_state,
                to_state=new_state,
                trigger=trigger,
                timestamp=time.time(),
                details=details,
            )
            self.history.append(transition)
            
            # Execute callbacks
            self._execute_callbacks(new_state)
            
            return new_state
        
        return self.state
    
    def _determine_new_state(self, trigger: TransitionTrigger,
                            details: Dict[str, Any]) -> Optional[AgentState]:
        """Determine new state based on trigger and current state."""
        
        # Success triggers → CHECKPOINT
        if trigger in [TransitionTrigger.TESTS_PASS, 
                       TransitionTrigger.PERFORMANCE_IMPROVED]:
            self._last_improvement_time = time.time()
            self.progress.consecutive_failures = 0
            return AgentState.CHECKPOINT
        
        # Checkpoint saved → back to AUTONOMOUS
        if trigger == TransitionTrigger.CHECKPOINT_SAVED:
            return AgentState.AUTONOMOUS
        
        # Intervention triggers → INTERVENE
        if trigger in [TransitionTrigger.TESTS_FAIL_REPEATED,
                       TransitionTrigger.PERFORMANCE_REGRESSION,
                       TransitionTrigger.MAJOR_CHANGE,
                       TransitionTrigger.USER_INTERRUPT,
                       TransitionTrigger.TIME_WITHOUT_PROGRESS,
                       TransitionTrigger.AGENT_UNCERTAIN,
                       TransitionTrigger.COST_LIMIT]:
            return AgentState.INTERVENE
        
        # Critical errors → ERROR
        if trigger in [TransitionTrigger.CRITICAL_ERROR,
                       TransitionTrigger.PROTECTED_FILE_TOUCHED]:
            return AgentState.ERROR
        
        # Completion triggers → COMPLETE
        if trigger in [TransitionTrigger.GOAL_REACHED,
                       TransitionTrigger.MAX_ITERATIONS,
                       TransitionTrigger.USER_QUIT]:
            return AgentState.COMPLETE
        
        return None
    
    def check_intervention_needed(self) -> Optional[TransitionTrigger]:
        """
        Check if intervention is needed based on policy and progress.
        
        Returns trigger if intervention needed, None otherwise.
        """
        if not self.policy:
            return None
        
        # Check consecutive failures
        if self.progress.consecutive_failures >= self.policy.consecutive_failures:
            return TransitionTrigger.TESTS_FAIL_REPEATED
        
        # Check performance regression
        if self.progress.best_latency_us < float('inf'):
            regression = (self.progress.current_latency_us - self.progress.best_latency_us) / self.progress.best_latency_us
            if regression * 100 > self.policy.performance_regression_pct:
                return TransitionTrigger.PERFORMANCE_REGRESSION
        
        # Check time without progress
        time_since = time.time() - self._last_improvement_time
        if time_since / 60 > self.policy.time_without_progress_mins:
            return TransitionTrigger.TIME_WITHOUT_PROGRESS
        
        return None
    
    def update_progress(self, **kwargs):
        """Update optimization progress."""
        for key, value in kwargs.items():
            if hasattr(self.progress, key):
                setattr(self.progress, key, value)
        
        # Update derived values
        if self.progress.baseline_latency_us > 0:
            self.progress.speedup_vs_baseline = (
                self.progress.baseline_latency_us / 
                max(self.progress.current_latency_us, 0.01)
            )
        
        if self.progress.best_latency_us < float('inf'):
            self.progress.speedup_vs_best = (
                self.progress.best_latency_us / 
                max(self.progress.current_latency_us, 0.01)
            )
        
        self.progress.time_since_improvement = time.time() - self._last_improvement_time
    
    def register_callback(self, state: AgentState, callback: Callable):
        """Register a callback for when entering a state."""
        if state not in self._callbacks:
            self._callbacks[state] = []
        self._callbacks[state].append(callback)
    
    def _execute_callbacks(self, state: AgentState):
        """Execute callbacks for a state."""
        for callback in self._callbacks.get(state, []):
            try:
                callback(self.progress)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "state": self.state.name,
            "iteration": self.progress.iteration,
            "baseline_us": self.progress.baseline_latency_us,
            "current_us": self.progress.current_latency_us,
            "best_us": self.progress.best_latency_us,
            "speedup": self.progress.speedup_vs_baseline,
            "consecutive_failures": self.progress.consecutive_failures,
            "time_since_improvement": self.progress.time_since_improvement,
            "transitions": len(self.history),
        }



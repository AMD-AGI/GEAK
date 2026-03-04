---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: Mesh_Query_Ray

## Variant Context
- Input semantic type: Ray-mesh intersection (ray tracing / collision detection)
- Datatype(s): fp32 (ray origin, direction, intersection parameters), int32 (face indices)
- Data representation: BVH-accelerated mesh with packed nodes, triangle soup
- Target architecture: gfx90a, gfx942 (AMD GPUs via HIP)

## Functionality
This kernel performs ray-mesh intersection queries using BVH traversal. For each ray, it traverses the BVH tree, tests ray-AABB intersection for internal nodes, and ray-triangle intersection for leaf nodes. Returns the closest intersection point with barycentric coordinates, face normal, and face index.

Two variants exist:
1. `mesh_query_ray`: Simple stack-based traversal
2. `mesh_query_ray_ordered`: Front-to-back ordered traversal with distance caching

---

## Optimization 1: Front-to-Back Ordered Traversal (mesh_query_ray_ordered)
- Commit ID: 18bf8c6 (initial), refined in 77d39fe, 386f991
- Optimization type: Compute / Memory
- Summary: Added ordered BVH traversal variant that processes closer nodes first for earlier termination

- Detailed explanation:
  The `mesh_query_ray_ordered` function implements front-to-back BVH traversal by:
  1. Maintaining a distance value for each stack entry
  2. Testing both children's AABB intersection distances
  3. Pushing the farther child first (so closer child is popped first)
  4. Early-exit when cached distance exceeds current best hit
  
  This ordering increases the probability of finding the closest hit early, allowing more aggressive pruning of the BVH tree.

- Code excerpt:
    ```cpp
    CUDA_CALLABLE inline bool mesh_query_ray_ordered(uint64_t id, const vec3& start, const vec3& dir, 
        float max_t, float& t, float& u, float& v, float& sign, vec3& normal, int& face)
    {
        Mesh mesh = mesh_get(id);

        int stack[BVH_QUERY_STACK_SIZE];
        float stack_dist[BVH_QUERY_STACK_SIZE];

        stack[0] = *mesh.bvh.root;
        stack_dist[0] = -FLT_MAX;
        int count = 1;

        while (count)
        {
            count -= 1;
            const int nodeIndex = stack[count];
            const float nodeDist = stack_dist[count];

            // Early exit using cached distance
            if (nodeDist < min_t)
            {
                // ... process node ...
                
                // For internal nodes: order children by distance
                float left_dist = FLT_MAX;
                bool left_hit = intersect_ray_aabb(..., left_dist);
                float right_dist = FLT_MAX;
                bool right_hit = intersect_ray_aabb(..., right_dist);

                // Push farther child first (closer child popped first)
                if (left_dist < right_dist)
                {
                    _swap(left_index, right_index);
                    _swap(left_dist, right_dist);
                    _swap(left_hit, right_hit);
                }

                if (left_hit && left_dist < min_t)
                {
                    stack[count] = left_index;
                    stack_dist[count] = left_dist;
                    count += 1;
                }

                if (right_hit && right_dist < min_t)
                {
                    stack[count] = right_index;
                    stack_dist[count] = right_dist;
                    count += 1;
                }
            }
        }
    }
    ```

- Evidence mapping:
  - Distance caching: `float stack_dist[BVH_QUERY_STACK_SIZE]` stores intersection distance per node
  - Early exit: `if (nodeDist < min_t)` skips nodes that can't improve result
  - Front-to-back ordering: `_swap()` ensures closer child is processed first
  - Pruning: `left_dist < min_t` check before pushing to stack

---

## Optimization 2: Optimized mesh_query_ray with Advanced Traversal (Experimental - Reverted)
- Commit ID: 5fd88ff (later reverted in a7a4407)
- Optimization type: Compute / Memory
- Summary: Attempted to add distance caching and front-to-back ordering to the main mesh_query_ray function

- Detailed explanation:
  This optimization attempted to apply the same front-to-back ordering strategy to the main `mesh_query_ray` function, along with:
  1. Precomputed ray shear coefficients for faster Woop intersection
  2. Fast triangle intersection without normal computation
  3. Deferred normal computation only for final hit
  
  However, this was reverted because the increased register pressure and code complexity did not yield performance improvements on AMD GPUs.

- Code excerpt (experimental - reverted):
    ```cpp
    // Precomputed ray shear coefficients for Woop intersection
    struct RayShear {
        int kx, ky, kz;
        float Sx, Sy, Sz;
    };

    // Short-stack with distance caching
    struct StackEntry {
        int node;
        float dist;
    };
    StackEntry stack[BVH_QUERY_STACK_SIZE];

    const RayShear rs = compute_ray_shear(dir);

    // Fast triangle intersection without normal
    if (intersect_ray_tri_woop_fast(start, rs, p, q, r, tri_t, tri_u, tri_v, tri_sign))
    {
        // ...
    }

    // Deferred normal computation for final hit only
    if (min_t < max_t)
    {
        const vec3 ab = q - p;
        const vec3 ac = r - p;
        normal = normalize(cross(ab, ac));
    }
    ```

- Evidence mapping:
  - Precomputed shear: `RayShear` struct computed once per ray
  - Fast intersection: `intersect_ray_tri_woop_fast` skips normal computation
  - Deferred normal: Normal computed only after finding closest hit
  - Note: Reverted due to no performance improvement on AMD GPUs

---

## Optimization 3: Deterministic Tie-Breaking (Experimental - Reverted)
- Commit ID: 18bf8c6 (later reverted in 79996ab)
- Optimization type: Robustness
- Summary: Added epsilon-based tie-breaking for nearly-equal intersection distances

- Detailed explanation:
  When multiple triangles have nearly equal intersection distances, the original code could produce non-deterministic results. This optimization added explicit tie-breaking: prefer front-facing triangles, then prefer smaller primitive indices.

- Code excerpt (experimental - reverted):
    ```cpp
    if (intersect_ray_tri_woop(start, dir, p, q, r, t, u, v, sign, &n))
    {
        const float tie_eps = 1e-6f * (1.0f + fabsf(min_t));
        bool accept = false;
        
        if (t < min_t - tie_eps)
        {
            accept = true;
        }
        else if (fabsf(t - min_t) <= tie_eps)
        {
            if (sign > min_sign)
                accept = true;
            else if (sign == min_sign && primitive_index < min_face)
                accept = true;
        }

        if (accept && t >= 0.0f)
        {
            min_t = t;
            min_face = primitive_index;
            // ...
        }
    }
    ```

- Evidence mapping:
  - Epsilon comparison: `tie_eps = 1e-6f * (1.0f + fabsf(min_t))`
  - Front-face preference: `sign > min_sign`
  - Index tie-break: `primitive_index < min_face`
  - Note: Reverted as overhead outweighed benefits

---

## Optimization 4: Stack Size and Epsilon Tuning (Experimental - Reverted)
- Commit ID: 77d39fe (later reverted in 386f991)
- Optimization type: Memory / Precision
- Summary: Attempted to reduce stack size and adjust AABB expansion epsilon

- Detailed explanation:
  This optimization attempted to:
  1. Reduce stack size from BVH_QUERY_STACK_SIZE (32) to 24 to reduce register pressure
  2. Reduce AABB expansion epsilon from 1e-3 to 1e-4 for tighter bounds
  
  However, these changes were reverted as they did not provide significant performance improvement.

- Code excerpt (experimental - reverted):
    ```cpp
    // Reduced stack size
    int stack[24];
    float stack_dist[24];

    // Tighter epsilon
    const float eps = 1.e-4f;
    ```

- Evidence mapping:
  - Smaller stack: `int stack[24]` vs `int stack[BVH_QUERY_STACK_SIZE]`
  - Tighter epsilon: `1.e-4f` vs `1.e-3f`
  - Note: Reverted to original values

---

## Optimization 5: Final State - Two Traversal Variants
- Commit ID: 386f991 (final)
- Optimization type: API Design
- Summary: Maintained two traversal variants for different use cases

- Detailed explanation:
  The final implementation provides two ray-mesh intersection functions:
  
  1. `mesh_query_ray`: Simple stack-based traversal without ordering
     - Lower register pressure
     - Simpler code path
     - Good for divergent ray patterns
  
  2. `mesh_query_ray_ordered`: Front-to-back ordered traversal
     - Distance caching for early termination
     - Better for coherent ray patterns
     - Uses `intersect_ray_tri_rtcd` instead of Woop algorithm

- Code excerpt (mesh_query_ray - simple):
    ```cpp
    CUDA_CALLABLE inline bool mesh_query_ray(uint64_t id, const vec3& start, const vec3& dir, 
        float max_t, float& t, float& u, float& v, float& sign, vec3& normal, int& face)
    {
        int stack[BVH_QUERY_STACK_SIZE];
        stack[0] = *mesh.bvh.root;
        int count = 1;

        while (count)
        {
            const int node_index = stack[--count];
            // ... simple traversal without distance caching ...
            if (hit && temp_t < min_t)
            {
                if (lower.b)
                {
                    // Test triangles using Woop algorithm
                    if (intersect_ray_tri_woop(start, dir, p, q, r, t, u, v, sign, &n))
                    // ...
                }
                else
                {
                    // Push both children without ordering
                    stack[count++] = lower.i;
                    stack[count++] = upper.i;
                }
            }
        }
    }
    ```

- Evidence mapping:
  - Two variants: `mesh_query_ray` and `mesh_query_ray_ordered`
  - Different algorithms: Woop vs RTCD for triangle intersection
  - Trade-off: Simplicity vs early termination optimization

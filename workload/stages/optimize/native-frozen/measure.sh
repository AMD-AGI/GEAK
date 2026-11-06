#!/usr/bin/env bash
# All native task methods use the optimize slot's single UT reader.
exec bash "$(dirname "${BASH_SOURCE[0]}")/../measure.sh"

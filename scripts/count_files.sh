#!/bin/sh

count=$(find . -type f -name '*.md' | wc -l | tr -d '[:space:]')
echo "Number of (*.md) Files: $count"
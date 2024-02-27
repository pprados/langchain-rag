#!/usr/bin/env bash
if [ `git rev-parse --abbrev-ref HEAD` == "master" ]; then
    make validate
fi

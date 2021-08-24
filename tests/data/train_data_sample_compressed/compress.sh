#!/usr/bin/bash

set -x
set -e

for i in *.csv ; do gzip ${i} ; done

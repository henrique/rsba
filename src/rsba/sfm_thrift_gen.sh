#!/bin/bash

rm -rf gen-cpp/
thrift --gen cpp sfm.thrift
ls -lR gen-cpp/
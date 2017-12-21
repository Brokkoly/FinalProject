#!/bin/sh
squeue>squeue.out
grep bkrull2 squeue.out
rm squeue.out



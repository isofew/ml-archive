#! /bin/bash

cd `dirname $0`

mkfifo qin qout
sh daemon.sh qin qout &
DAEMON=$!

function cleanup {
    rm qin qout
    kill -s 9 $DAEMON
}

trap cleanup EXIT

python $@

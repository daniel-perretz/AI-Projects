#!/bin/bash
name="main"
command="python3 $name.py"
error_log_file="$name-error.log"

while true; do
   if ! pgrep -f "$command" > /dev/null; then
        echo "[$(date)] The python script stopped running. Restarting now..." >> "$error_log_file"
        $command &
   fi
   sleep 60
done

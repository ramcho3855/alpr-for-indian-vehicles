#!/bin/bash

set -e

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pwkkDjcS7uVsvC0wucIgzXLzj0o2bDl7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1pwkkDjcS7uVsvC0wucIgzXLzj0o2bDl7" -O data.zip && rm -rf /tmp/cookies.txt
unzip data.zip
rm data.zip


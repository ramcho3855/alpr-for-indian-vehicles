#!/bin/bash

set -e


wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1K9Q044bCB7-O7LtrY7UrSzF2V4lmDS8D' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1K9Q044bCB7-O7LtrY7UrSzF2V4lmDS8D" -O data.zip && rm -rf /tmp/cookies.txt
unzip data.zip
rm data.zip


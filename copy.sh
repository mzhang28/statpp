#!/usr/bin/env bash
F=$(mktemp)
echo -e "I'm writing an unsupervised EM-style osu! performance system.\n" >> $F
echo -e "Here are the design goals:\n" >> $F
echo -e "--- begin README.md ---\n" >> $F
cat README.md >> $F
echo -e "\n--- end README.md ---\n" >> $F

echo -e "Here is the source code:\n" >> $F

echo -e "--- begin db.py ---\n" >> $F
cat db.py >> $F
echo -e "\n--- end db.py ---\n" >> $F

echo -e "--- begin tunable.py ---\n" >> $F
cat tunable.py >> $F
echo -e "\n--- end tunable.py ---\n" >> $F

echo -e "--- begin main.py ---\n" >> $F
cat main.py >> $F
echo -e "\n--- end main.py ---\n" >> $F

cat $F | pbcopy

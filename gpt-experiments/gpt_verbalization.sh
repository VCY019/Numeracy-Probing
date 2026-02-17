#!/bin/bash

# Reproduces the output files in verbalization-test/ (filenames use uppercase GPT-4.1 / GPT-4.1-mini).
# Run this script from the gpt-experiments/ directory with OPENAI_API_KEY set in your environment.

# for Figure 11

perl -ne 'm/"a": "(.*?)".*"b": "(.*?)"/ || die "bad format"; print "$1\t$2\n"' ../data/dec_sci_compare/test.jsonl > dec_sci_compare_prompts.txt
time ./which-is-larger.py dec_sci_compare_prompts.txt --model gpt-4.1 --sysprompt 'Just answer with a number.' > dec_sci_compare_output_gpt-4.1.txt
time ./which-is-larger.py dec_sci_compare_prompts.txt --model gpt-4.1-mini --sysprompt 'Just answer with a number.' > dec_sci_compare_output_gpt-4.1-mini.txt

# other half of Figure 11 - uses the same dec_sci_compare_prompts.txt
time ./which-is-larger.py dec_sci_compare_prompts.txt --model gpt-4.1 --sysprompt 'Just answer with a number.' --reverse > dec_sci_compare_output_gpt-4.1_alt.txt
time ./which-is-larger.py dec_sci_compare_prompts.txt --model gpt-4.1-mini --sysprompt 'Just answer with a number.' --reverse > dec_sci_compare_output_gpt-4.1-mini_alt.txt



# for Table 1

perl -ne 'm/"a": "(.*?)".*"b": "(.*?)"/ || die "bad format"; print "$1\t$2\n"' ../data/int_sci_compare/test.jsonl > int_sci_compare_prompts.txt
time ./which-is-larger-shots.py 1 int_sci_compare_prompts.txt --model gpt-4.1 --sysprompt 'Just answer with a number.' > int_sci_compare_output_gpt-4.1.txt
time ./which-is-larger-shots.py 1 int_sci_compare_prompts.txt --model gpt-4.1-mini --sysprompt 'Just answer with a number.' > int_sci_compare_output_gpt-4.1-mini.txt
# for Figure 12
# uses the same int_sci_compare_prompts.txt that was generated above
# runs in parallel

for n in 2 3 4 5; do
   for model in gpt-4.1 gpt-4.1-mini; do
      time ./which-is-larger-shots.py $n int_sci_compare_prompts.txt --model $model --sysprompt 'Just answer with a number.' > int_sci_compare_output_${model}_${n}shot.txt &
   done
done

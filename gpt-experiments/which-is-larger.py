#!/usr/bin/env python

import os
from openai import OpenAI
from argparse import ArgumentParser, FileType
import json

parser = ArgumentParser()
parser.add_argument("file", help="two-column flat with example (default: stdin)", nargs="?", default="-", type=FileType('r'))
parser.add_argument("--model", help="which OpenAI model to use", default="gpt-4.1")
parser.add_argument("--sysprompt", help="OpenAI system instructions (default: none)")
parser.add_argument("--reverse", help="reverse order of numbers in few-shot example", action='store_true')

args = parser.parse_args()

api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found. Set the environment variable OPENAI_API_KEY with your API key.")
client = OpenAI(api_key=api_key)

prompt = [  { 'role': 'user', 
              'content':
              u'Which is larger, 899.9 or 9.9 × 10^2?' if args.reverse else u'Which is larger, 9.9 × 10^2 or 899.9?' },
            { 'role': 'assistant',
              'content': u'9.9 × 10^2' },
            { 'role': 'user', 
              'content': 'Which is larger, {} or {}?' } ]

if args.sysprompt is not None:
    prompt.insert(0, { 'role': 'system', 'content': args.sysprompt })
    
# would be faster and cheaper to use the batch API, but this is a small job
# and doing it sequentially saves us the step of sorting the results
for line in args.file:
    a,b = line.rstrip('\n').split("\t")   # the two numbers to compare
    template = prompt[-1]['content']
    prompt[-1]['content'] = template.format(a,b)  # destructively interpolate into template
    response = client.chat.completions.create(model=args.model,
                                              temperature=0,
                                              messages=prompt)
    prompt[-1]['content'] = template   # undo destructive interpolation
    print(response.choices[0].message.content.replace('\n','\\n'))

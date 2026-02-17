#!/usr/bin/env python

import os
from openai import OpenAI
from argparse import ArgumentParser, FileType
import json

parser = ArgumentParser()
parser.add_argument("n", help="number of examples to use for few-shot prompt", type=int)
parser.add_argument("file", help="two-column flat with example (default: stdin)", nargs="?", default="-", type=FileType('r'))
parser.add_argument("--model", help="which OpenAI model to use", default="gpt-4.1")
parser.add_argument("--sysprompt", help="OpenAI system instructions (default: none)")

args = parser.parse_args()

api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found. Set the environment variable OPENAI_API_KEY with your API key.")
client = OpenAI(api_key=api_key)

prompt = [  { 'role': 'user', 
              'content': u'Which is larger, 9.9 × 10^2 or 100?' }, # example 1
            { 'role': 'assistant',
              'content': u'9.9 × 10^2' },
            { 'role': 'user', 
              'content': u'Which is larger,161230 or 7.182 × 10^5 ?' }, # example 2
            { 'role': 'assistant',
              'content': u'7.182 × 10^5' },
            { 'role': 'user', 
              'content': u'Which is larger, 713 or 4.78 × 10^2?' }, # example 3
            { 'role': 'assistant',
              'content': u'713' }, 
            { 'role': 'user', 
              'content': u'Which is larger, 1.354 × 10^6 or 4906723?' },  # example 4
            { 'role': 'assistant',
              'content': u'4906723' },
            { 'role': 'user', 
              'content': u'20834 or 6.5 × 10^3' },  # example 5
            { 'role': 'assistant',
              'content': u'20834' },
            { 'role': 'user', 
              'content': 'Which is larger, {} or {}?' } ]

# Keep only the first n shots from the prompt, and the final question
if 2*args.n < 0 or 2*args.n >= len(prompt):
    raise ValueError("n argument is out of range")
prompt = prompt[:(2*args.n)] + prompt[-1:]  

# Prepend the system prompt, if any
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

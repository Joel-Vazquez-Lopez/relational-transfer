#!/usr/bin/env python3
"""Exact paired binary-outcome sensitivity from aggregate margins only."""
import math,json

def exact_p(x,y):
 n=x+y
 return min(1.0,2*sum(math.comb(n,k) for k in range(min(x,y)+1))/2**n) if n else 1.0

def tables(n,a,b):
 if not (0<=a<=n and 0<=b<=n): raise ValueError("invalid margins")
 return [{"both":z,"a_only":a-z,"b_only":b-z,"neither":n-a-b+z,
          "discordant":a+b-2*z,"exact_p":exact_p(a-z,b-z)}
         for z in range(max(0,a+b-n), min(a,b)+1)]

def summarize(n,a,b):
 t=tables(n,a,b)
 return {"n":n,"a":a,"b":b,"minimum_discordant":min(x["discordant"] for x in t),
         "maximum_discordant":max(x["discordant"] for x in t),
         "possible_exact_p_range":[min(x["exact_p"] for x in t),
                                   max(x["exact_p"] for x in t)],
         "all_question_level_p_below_005":all(x["exact_p"]<0.05 for x in t),
         "all_tables":t}

if __name__=="__main__":
 print(json.dumps({
  "reference_vs_swapped_accuracy":summarize(48,34,26),
  "fresh_vs_transcript_accuracy":summarize(48,26,7),
  "fresh_vs_transcript_format_compliance":summarize(48,42,18)},indent=2))

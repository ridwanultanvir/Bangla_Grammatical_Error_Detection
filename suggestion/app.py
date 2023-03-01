# https://github.com/srajanseth84/FTG/blob/main/app.py


import gradio as gr
from transformers import pipeline
from normalizer import normalize
import re
from Levenshtein import distance

# Load the bert model
fill_mask = pipeline(
  "fill-mask", 
  model="csebuetnlp/banglabert", 
  tokenizer="csebuetnlp/banglabert",
  device=-1,
  top_k=32000
)


def fill_mask_wrapped(text):
  text = normalize(text)
  # Extract spans with $$ with regex
  spans = re.findall(r'\$.*?\$', text)

  # Extract words from the spans list from the text
  spans_words = [re.sub(r'\$','', span) for span in spans]
  print(spans_words)


  # Replace the spans with [MASK] token  
  for span in spans:
    text = text.replace(span, '[MASK]')
  
  output = fill_mask(text)
  outputs = []
  for out in output:
    token =  out['token_str']
    score = out['score']
    distance_score = -distance(token, spans_words[0])
    if distance_score != 0:
      outputs.append((token, score, distance_score))
  
  # Sort first by score and then by distance
  outputs.sort(key=lambda x: (x[2], x[1]), reverse=True)

  # Replace the [MASK] token with the first output
  text2 = text.replace('[MASK]', outputs[0][0])
  text3 = text.replace('[MASK]', outputs[1][0])
  text4 = text.replace('[MASK]', outputs[2][0])


  # import pdb; pdb.set_trace()

  # print(output)
  # return "Hi"
  print(text2, text3, text4)
  return text2, text3, text4


# Create a gradio app with bert model for mask filling
app = gr.Interface(
  fn=fill_mask_wrapped, 
  inputs="text", 
  outputs= ["text", "text", "text"],
  title="Mask Filling", 
  description="Fill the mask with the correct word")

# Run the app
app.launch()
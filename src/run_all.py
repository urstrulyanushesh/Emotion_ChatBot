#!/usr/bin/env python
# coding: utf-8

# # ▶️ run_all.ipynb — Master Controller
# **Runs all notebooks in order automatically.**
# 
# Just run this single cell to execute the entire pipeline!
# 
# ```
# 01_data_collection   → saves emotion_data.csv
# 02_preprocessing     → saves cleaned_data.csv
# 03_eda               → shows charts
# 04_feature_engineering → saves tfidf + splits
# 05_model_training    → saves emotion_model.pkl
# 06_deployment        → tests chatbot
# 07_testing           → alpha & beta tests
# 08_optimized_app     → launches Gradio UI
# ```

# In[ ]:


# get_ipython().run_line_magic('pip', 'install nbformat nbconvert -q')


# In[ ]:


import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import time

notebooks = [
    '01_data_collection.ipynb',
    '02_preprocessing.ipynb',
    '03_eda.ipynb',
    '04_feature_engineering.ipynb',
    '05_model_training.ipynb',
    '06_deployment.ipynb',
    '07_testing.ipynb',
    # '08_optimized_app.ipynb',   # ← uncomment to auto-launch Gradio at end
]

print('🚀 Running full ML pipeline...\n')
print('='*50)

total_start = time.time()
failed = []

# for nb_name in notebooks:
#     start = time.time()
#     print(f'▶️  {nb_name}...')
#     try:
#         # # with open(nb_name) as f:
#         #     nb = nbformat.read(f, as_version=4)

#         ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
#         ep.preprocess(nb, {'metadata': {'path': '.'}})
#         elapsed = time.time() - start
#         print(f'   ✅ Done in {elapsed:.1f}s\n')
#     except Exception as e:
#         print(f'   ❌ FAILED: {e}\n')
#         failed.append(nb_name)

for nb_name in notebooks:
    start = time.time()
    print(f'▶️  {nb_name}...')
    try:
        # 1. We MUST open the file first and tell it to use UTF-8 (to fix the emoji/codec error)
        with open(nb_name, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # 2. Now that 'nb' is loaded, we can process it
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        
        elapsed = time.time() - start
        print(f'   ✅ Done in {elapsed:.1f}s\n')
    except Exception as e:
        print(f'   ❌ FAILED: {e}\n')
        failed.append(nb_name)



total = time.time() - total_start
print('='*50)
print(f'\n🎉 Pipeline complete in {total:.1f}s')
if failed:
    print(f'⚠️  Failed notebooks: {failed}')
else:
    print('✅ All notebooks ran successfully!')
    print('\nNow run 08_optimized_app.ipynb to launch the chatbot! 🤖')


# In[ ]:


import sys
print(sys.executable)


# In[ ]:





# CoWAR：A General Complementary Web API Recommendation Framework based on Learning Model

## Introuction

CoWAR is designed to recommend complementary Web APIs tailored for Mashup creation, based on the user’s selected Web APIs.



## Environment Requirement

The experiment was run in the python 3.10.0 environment, and the required packages are as follows：

- wheel == 0.37.1
- transformers == 4.31.0
- torch == 2.0.1
- numpy == 1.24.3
- nlp == 0.4.0
- pandas == 2.0.3
- scikit-learn == 1.2.2
- matplotlib == 3.7.1
- scipy == 1.11.1
- tokenizers == 0.13.3
- seaborn == 0.12.2



## Example to run CoWAR

1. Get the Sentece-BERT file from Hugging Face, the version of sbert is **all-mpnet-base-v2**, and place it in the **sbert folder**
2. Run  **sample_generation/api_representation.py** first to get the input data
3. Then run **experiments/experiment_sanfm.py**



## Contact

Email：1302466947@qq.com / qiqichen0702@gmail.com (Qiqi Chen)


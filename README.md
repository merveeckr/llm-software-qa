# BERT Gereksinim Analizi Uygulaması

Bu proje, Türkçe ve İngilizce gereksinim verilerini işlemek için BERT tabanlı modeller kullanan bir uygulamadır. Aşağıda proje dosyaları, gereksinimler ve kullanım talimatları detaylı olarak açıklanmıştır.

---

## Gereksinimler

- Python 3.12.10 veya 3.9  
> Not: Diğer Python sürümlerinde `transformers` kütüphanesi uyumsuz olabilir.

- Gerekli Python kütüphanelerini yüklemek için:
```bash
pip install -r req.txt
```

- Python 3.12.10 veya 3.9 kurunuz diğer modellerde transformers kütüphanesinin sürümü çalışmayacaktır.

- app.py dosyasını çalıştırmak için 
```bash
python -m streamlit run app.py 
```

- app.py -> asıl uygulamamızı ayağa kaldırcak olan dosyadır.

- bil.py -> bertürkün eğitimini sağlayan dosyadır.
 ```bash
python bil.py
```

- eng.py -> bertin eğitimini gerçekleştiren dosyadır 
 ```bash
python eng.py
```

- english.py -> eğitimi biten bertin app.py dosyasında çalıştırılmasını sağlar. 

- eng.csv -> gereksinim datasının ingilizce verisyonu 
- yeni.csv -> türkçe gereksinim datası 

- short_eng.csv ve short_türk.csv test amaçlı yazılmış kısa datasetler. 

## Kullanılan LLM modelleri 
- bert ingilizce -> https://huggingface.co/google-bert/bert-base-uncased
- bert türkçe -> https://huggingface.co/dbmdz/bert-base-turkish-cased
- gemma3-1b -> https://huggingface.co/google/gemma-3-1b-it
- llama3.1-8b -> https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct


## Önerilen Akış

1. Önce model eğitimini (bil.py ve eng.py) gerçekleştirin.

2. Eğitilmiş modeli test etmek için app.py çalıştırın.

3. Analiz ve görselleştirmeleri Streamlit üzerinden görüntüleyin.

------------------------------------------------------------------------
- contact: Merve ÇAKIR / cakirmerve1629@gmail.com
------------------------------------------------------------------------







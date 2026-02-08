# RAG აგენტი - საგადასახადო და საბაჟო ჰაბი

RAG (Retrieval-Augmented Generation) აგენტი, რომელიც პასუხობს კითხვებს ქართულ ენაზე საგადასახადო და საბაჟო ადმინისტრირების შესახებ, ყოველთვის მითითებული წყაროთი.

**წყარო:** საინფორმაციო და მეთოდოლოგიურ ჰაბი (საგადასახადო და საბაჟო ადმინისტრირების შესახებ დოკუმენტები და ინფორმაცია ერთ სივრცეში) - https://infohub.rs.ge/ka

## მახასიათებლები

- ✅ პასუხები **ქართულ ენაზე**
- ✅ **ყოველთვის** წყაროს მითითება მოცემული ფორმატით
- ✅ მულტიბილინგვალური embeddings (Georgian support)
- ✅ FAISS ვექტორული მაღაზია
- ✅ Streamlit დემო ინტერფეისი

## სწრაფი დაწყება

### 1.
```bash
pip install -r requirements.txt
```

### 2. OpenAI API გასაღები (LLM-ისთვის)

```bash
set OPENAI_API_KEY=your-api-key-here
```

ან შექმენით `.env` ფაილი:
```
OPENAI_API_KEY=your-api-key-here
```

### 3. დოკუმენტების დამატება

მოათავსეთ PDF, DOCX ან TXT ფაილები `data/` საქაღალდეში. ნიმუში დოკუმენტი უკვე შედის.

### 4. აპლიკაციის გაშვება

```bash
streamlit run src/app.py
```

ან:

```bash
python run.py
```

ბრაუზერში გაიხსნება http://localhost:8501

## პროექტის სტრუქტურა

```
rag-agent/
├── data/                 # დოკუმენტები (PDF, DOCX, TXT)
├── src/
│   ├── app.py           # Streamlit UI
│   ├── config.py        # კონფიგურაცია და citation
│   ├── embedder.py      # Multilingual embeddings
│   ├── loader.py        # დოკუმენტების ჩატვირთვა
│   ├── rag_pipeline.py  # RAG ჯაჭვი
│   └── vectorstore.py   # FAISS vector store
├── requirements.txt
└── README.md
```

## GitHub და დემო ლინკები

- **GitHub:** [https://github.com/Predaotor/RAG-](https://github.com/Predaotor/RAG-)
- **დემო:** [Streamlit Cloud-ზე ]()

## წყაროს მითითება

ყოველი პასუხი მთავრდება:

> საინფორმაციო და მეთოდოლოგიურ ჰაბზე გამოქვეყნებული დოკუმენტების მიხედვით (საგადასახადო და საბაჟო ადმინისტრირების შესახებ დოკუმენტები და ინფორმაცია ერთ სივრცეში) - https://infohub.rs.ge/ka

## ლიცენზია

MIT

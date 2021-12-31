import streamlit as st
from transformers import PegasusTokenizer
import torch
import pandas as pd
from googletrans import Translator
translator = Translator(service_urls=[
      'translate.google.com',
      'translate.google.co.kr',
    ])
LANGUAGES = {
    'Afrikaans':'af','Albanian':'sq','Amharic': 'am','Arabic':'ar' ,'Armenian':'hy','Azerbaijani':'az','Basque':'eu','Belarusian':'be','Bengali':'bn',
    'Bosnian':'bs','Bulgarian':'bg','Catalan':'ca','Cebuano':'ceb','Chichewa':'ny','Chinese (simplified)':'zh-cn','Chinese (traditional)':'zh-tw','Corsican':'co','Croatian':'hr',
    'Czech':'cs','Danish':'da','Dutch':'nl','English':'en','Esperanto':'eo','Estonian':'et','Filipino':'tl','Finnish':'fi',
    'French':'fr','Frisian':'fy','Galician':'gl','Georgian':'ka','German':'de','Greek':'el','Gujarati':'gu','Haitian creole':'ht',
    'Hausa':'ha','Hawaiian':'haw','Hebrew':'he','Hindi':'hi','Hmong':'hmn','Hungarian':'hu','Icelandic':'is',
    'Igbo':'ig','Indonesian':'id','Irish':'ga','Italian':'it','Japanese':'ja','Javanese':'jw','Kannada':'kn','Kazakh':'kk','Khmer':'km',
    'Korean':'ko','Kurdish (kurmanji)':'ku','Kyrgyz':'ky','Lao':'lo','Latin':'la','Latvian':'lv','Lithuanian':'lt','Luxembourgish':'lb',
    'Macedonian':'mk','Malagasy':'mg','Malay':'ms','Malayalam':'ml','Maltese':'mt','Maori':'mi','Marathi':'mr','Mongolian':'mn',
    'Myanmar (burmese)':'my','Nepali':'ne','Norwegian':'no','Odia':'or','Pashto':'ps','Persian':'fa','Polish':'pl','Portuguese':'pt',
    'Punjabi':'pa','Romanian':'ro','Russian':'ru','Samoan':'sm','Scots gaelic':'gd','Serbian':'sr','Sesotho':'st','Shona':'sn','Sindhi':'sd',
    'Sinhala':'si','Slovak':'sk','Slovenian':'sl','Somali':'so','Spanish':'es','Sundanese':'su','Swahili':'sw', 'Swedish':'sv','Tajik':'tg',
    'Tamil':'ta','Telugu':'te','Thai':'th','Turkish':'tr','Ukrainian':'uk','Urdu':'ur','Uyghur':'ug','Uzbek':'uz','Vietnamese':'vi','Welsh':'cy',
    'Xhosa':'xh','Yiddish':'yi','Yoruba':'yo','Zulu':'zu'}
h=st.sidebar.selectbox("Select Activity",["Translator and Summarizer","Pegasus Tokenizer"])
try:
    if h=="Translator and Summarizer" :
        st.subheader("LANGUAGE TRANSLATOR AND SUMMARZIER")
        col1,col2=st.columns(2)
        input1=col1.selectbox("Source Language",['Auto','Afrikaans','Albanian','Amharic','Arabic','Armenian','Azerbaijani','Aasque','Belarusian','Bengali','Bosnian','Bulgarian','Catalan', 'Cebuano','Chichewa','Chinese (simplified)','Chinese (traditional)','Corsican',
    'Croatian','Czech','Danish','Dutch','English','Esperanto','Estonian','Filipino', 'Finnish', 'French','Frisian','Galician','Georgian', 'German','Greek','Gujarati','Haitian creole',
    'Hausa','Hawaiian','Hebrew', 'Hebrew', 'Hindi','Hmong','Hungarian','Icelandic','Igbo','Indonesian','Irish','Italian','Japanese','Javanese','Kannada','Kazakh','Khmer','Korean',
    'Kurdish (kurmanji)','Kyrgyz','Lao','Latin','Latvian', 'Lithuanian','Luxembourgish','Macedonian','Malagasy','Malay','Malayalam','Maltese','Maori','Marathi','Mongolian','Myanmar (burmese)','Nepali','Norwegian',
    'Odia','Pashto','Persian','Polish','Portuguese','Punjabi','Romanian','Russian','Samoan','Scots gaelic', 'Serbian','Sesotho','Shona','Sindhi','Sinhala','Slovak','Slovenian', 'Somali','Spanish','Sundanese', 'Swahili',   'Swedish',  'Tajik',
    'Tamil',  'Telugu','Thai', 'Turkish','Ukrainian', 'Urdu', 'Uyghur', 'Uzbek', 'Vietnamese',  'Welsh','Xhosa', 'Yiddish', 'Yoruba', 'Zulu'])
        input2=col2.selectbox("Destination Language",["English"])
        col3,col4=st.columns(2)
        input3=col3.selectbox("Translator",["Yes","No"])
        input4=col4.selectbox("Summarizer",["Yes","No"])
        input5= st.text_area("Paste Text Here",height=150)
        input6=st.button('Convert')
        if input6:
            if len(input5)>100:
                try:
                    if input3=="Yes" and input4=="No":
                        if input1=="Auto":
                            translated=translator.translate(input5, dest='en')
                            f=translated.text
                            st.success(f)
                        else:
                            language=LANGUAGES[input1]
                            translated = translator.translate(input5,src=language,dest='en')
                            f=translated.text
                            st.success(f)

                    elif input3=="Yes" and input4=="Yes":
                        if input1=="Auto":
                            path = "./model/lst.pth"
                            col3, col4 = st.columns(2)
                            translated = translator.translate(input5, dest='en')
                            f=translated.text
                            tokenizer = PegasusTokenizer.from_pretrained("./model/")
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            model = torch.load(path)
                            batch = tokenizer(f, truncation=True, padding='longest', return_tensors="pt").to(device)
                            translated = model.generate(**batch)
                            tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
                            col3_expander = col3.expander("Expand Translated Text")
                            with col3_expander:
                                col3_expander.write(f)
                            col4_expander = col4.expander("Expand Summarized Text")
                            with col4_expander:
                                col4_expander.write(tgt_text[0])
                        else:
                            language = LANGUAGES[input1]
                            path = "./model/lst.pth"
                            col3, col4 = st.columns(2)
                            translated = translator.translate(input5,src=language,dest='en')
                            f = translated.text
                            tokenizer = PegasusTokenizer.from_pretrained("./model/")
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            model = torch.load(path)
                            batch = tokenizer(f, truncation=True, padding='longest', return_tensors="pt").to(device)
                            translated = model.generate(**batch)
                            tgt_text = tokenizer.batch_decode(f, skip_special_tokens=True)
                            col3_expander = col3.expander("Expand Translated Text")
                            with col3_expander:
                                col3_expander.write(f)
                            col4_expander = col4.expander("Expand Summarized Text")
                            with col4_expander:
                                col4_expander.write(tgt_text[0])

                    elif input3=="No" and input4=="Yes" and input1=="Auto" or input1!="Auto":
                        if input1=="Auto":
                            path = "./model/lst.pth"
                            translated = translator.translate(input5, dest='en')
                            f=translated.text
                            tokenizer = PegasusTokenizer.from_pretrained("./model/")
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            model = torch.load(path)
                            batch = tokenizer(f, truncation=True, padding='longest', return_tensors="pt").to(device)
                            translated = model.generate(**batch)
                            tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
                            st.success(tgt_text[0])
                        else:
                            language = LANGUAGES[input1]
                            path = "./model/lst.pth"
                            translated = translator.translate(input5,src=language,dest='en')
                            f=translated.text
                            tokenizer = PegasusTokenizer.from_pretrained("./model/")
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            model = torch.load(path)
                            batch = tokenizer(f, truncation=True, padding='longest',
                                              return_tensors="pt").to(device)
                            translated = model.generate(**batch)
                            tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
                            st.success(tgt_text[0])

                    elif input3=="No" and input4=="No" and input1=="Auto" or input1!="Auto":
                        st.success("No Translator and summarizer is selected")
                except ValueError:
                    st.success("You have entered an incorrect language syntax")
            else:
                st.success("No of words in the sentence should be greater than 100")
    elif h== "Pegasus Tokenizer":
        st.subheader("PEGASUS TOKENIZER")
        user_input=st.text_area("Enter the Text")
        button1 = st.button("Predict")
        if button1:
            tokenizer2 = PegasusTokenizer.from_pretrained("./model/")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            inputs = tokenizer2(user_input, truncation=True, padding='longest', return_tensors="pt").to(device)
            j = pd.DataFrame()
            k = []
            for g in inputs["input_ids"][0]:
                k.append(tokenizer2.decode(g, skip_special_tokens=False, clean_up_tokenization_spaces=False))
            j["Text"]=k
            j["Tensor"] = inputs["input_ids"][0]
            st.write(j)
except ValueError:
    st.error("Please enter a valid input")

import streamlit as st
from googletrans import Translator
from transformers import PegasusTokenizer,PegasusForConditionalGeneration
from deep_translator import GoogleTranslator
import torch
import pandas as pd
LANGUAGES = {
    'afrikaans':'af','albanian':'sq','amharic': 'am','arabic':'ar' ,'armenian':'hy','azerbaijani':'az','basque':'eu','belarusian':'be','bengali':'bn',
    'bosnian':'bs','bulgarian':'bg','catalan':'ca','cebuano':'ceb','chichewa':'ny','chinese (simplified)':'zh-cn','chinese (traditional)':'zh-tw','corsican':'co','croatian':'hr',
    'czech':'cs','danish':'da','dutch':'nl','english':'en','esperanto':'eo','estonian':'et','filipino':'tl','finnish':'fi',
    'french':'fr','frisian':'fy','galician':'gl','georgian':'ka','german':'de','greek':'el','gujarati':'gu','haitian creole':'ht',
    'hausa':'ha','hawaiian':'haw','hebrew':'he','hindi':'hi','hmong':'hmn','hungarian':'hu','icelandic':'is',
    'igbo':'ig','indonesian':'id','irish':'ga','italian':'it','japanese':'ja','javanese':'jw','kannada':'kn','kazakh':'kk','khmer':'km',
    'korean':'ko','kurdish (kurmanji)':'ku','kyrgyz':'ky','lao':'lo','latin':'la','latvian':'lv','lithuanian':'lt','luxembourgish':'lb',
    'macedonian':'mk','malagasy':'mg','malay':'ms','malayalam':'ml','maltese':'mt','maori':'mi','marathi':'mr','mongolian':'mn',
    'myanmar (burmese)':'my','nepali':'ne','norwegian':'no','odia':'or','pashto':'ps','persian':'fa','polish':'pl','portuguese':'pt',
    'punjabi':'pa','romanian':'ro','russian':'ru','samoan':'sm','scots gaelic':'gd','serbian':'sr','sesotho':'st','shona':'sn','sindhi':'sd',
    'sinhala':'si','slovak':'sk','slovenian':'sl','somali':'so','spanish':'es','sundanese':'su','swahili':'sw', 'swedish':'sv','tajik':'tg',
    'tamil':'ta','telugu':'te','thai':'th','turkish':'tr','ukrainian':'uk','urdu':'ur','uyghur':'ug','uzbek':'uz','vietnamese':'vi','welsh':'cy',
    'xhosa':'xh','yiddish':'yi','yoruba':'yo','zulu':'zu'}
h=st.sidebar.selectbox("Select Activity",["Translator and Summarizer","Pegasus Tokenizer"])
try:
    if h=="Translator and Summarizer" :
        st.title("Language Translator and Summarizer")
        col1,col2=st.columns(2)
        input1=col1.selectbox("Source Language",['auto','afrikaans','albanian','amharic','arabic','armenian','azerbaijani','basque','belarusian','bengali','bosnian','bulgarian','catalan', 'cebuano','chichewa','chinese (simplified)','chinese (traditional)','corsican',
    'croatian','czech','danish','dutch','english','esperanto','estonian','filipino', 'finnish', 'french','frisian','galician','georgian', 'german','greek','gujarati','haitian creole',
    'hausa','hawaiian','hebrew', 'hebrew', 'hindi','hmong','hungarian','icelandic','igbo','indonesian','irish','italian','japanese','javanese','kannada','kazakh','khmer','korean',
    'kurdish (kurmanji)','kyrgyz','lao','latin','latvian', 'lithuanian','luxembourgish','macedonian','malagasy','malay','malayalam','maltese','maori','marathi','mongolian','myanmar (burmese)','nepali','norwegian',
    'odia','pashto','persian','polish','portuguese','punjabi','romanian','russian','samoan','scots gaelic', 'serbian','sesotho','shona','sindhi','sinhala','slovak','slovenian', 'somali','spanish','sundanese', 'swahili',   'swedish',  'tajik',
    'tamil',  'telugu','thai', 'turkish','ukrainian', 'urdu', 'uyghur', 'uzbek', 'vietnamese',  'welsh','xhosa', 'yiddish', 'yoruba', 'zulu'])
        input2=col2.selectbox("Destination Language",["English"])
        col3,col4=st.columns(2)
        input3=col3.selectbox("Translator",["Yes","No"])
        input4=col4.selectbox("Summarizer",["Yes","No"])
        input5= st.text_area("Paste Text Here",height=150)
        input6=st.button('Convert')
        if input6:
            if len(input5)>100:
                if input3=="Yes" and input4=="No" and input1=="auto":
                    translated = GoogleTranslator(source="auto", target="en")
                    f = translated.translate(input5)
                    print(f)
                    st.success(f)
                elif input3=="Yes" and input4=="Yes" and input1!="auto":
                    language=LANGUAGES[input1]
                    path = "./model/lst.pth"
                    col3, col4 = st.columns(2)
                    translated = GoogleTranslator(source=input1, target="en")
                    f = translated.translate(input5)
                    print(f)
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
                elif input3=="No" and input4=="Yes" and input1=="auto" or input1!="auto":
                    path = "./model/lst.pth"
                    translated = GoogleTranslator(source="auto", target="en")
                    f = translated.translate(input5)
                    print(f)
                    tokenizer = PegasusTokenizer.from_pretrained("./model/")
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model = torch.load(path)
                    batch = tokenizer(f, truncation=True, padding='longest', return_tensors="pt").to(device)
                    translated = model.generate(**batch)
                    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
                    st.success(tgt_text[0])
                elif input3=="No" and input4=="No" and input1=="auto" or input1!="auto":
                    st.success("No Translator and summarizer is selected")
            else:
                st.success("No of words in the sentence should be greater than 100")
    elif h== "Pegasus Tokenizer":
        st.title("Pegasus Tokenizer")
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

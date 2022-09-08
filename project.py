#importing necessary modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus  import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import sys
from tkinter import *
from tkinter import simpledialog
from tkinter import messagebox
ud7='y'
while(True):
    if(ud7=='y'):
        ud1=[]
        ud2=[]
        ud3=[]
        ud4=[]
        ud5=[]
        ud6=[]
        def get_me1():
            A=simpledialog.askstring('input','enter symptom1:')
            ud1.append(A)
        def get_me2():
            B=simpledialog.askstring('input','enter symptom2:')
            ud2.append(B)
        def get_me3():
            C=simpledialog.askstring('input','enter symptom3:')
            ud3.append(C)
        def get_me4():
            D=simpledialog.askstring('input','enter symptom4:')
            ud4.append(D)
        def get_me5():
            E=simpledialog.askstring('input','enter symptom5:')
            ud5.append(E)
        def get_me6():
            F=simpledialog.askstring('input','enter symptom6:')
            ud6.append(F)
        def call_me():
            ans=messagebox.askquestion("exit","Do you really want to exit")
            if(ans=='yes'):
                root.destroy()


        root = Tk()
        frame=Frame(root)
        label1=Label(frame,text="you have to enter 6 relevant symptoms if your input is not relevant than machine do not predict right disease",bg='cyan')
        label1.pack()
        label2=Label(frame,text="Click on the buttons mentioned below to enter your symptoms respectively",bg='yellow')
        label2.pack()
        label3=Label(frame,text="After entering all symptoms you can press exit for getting diseases.",bg='red')
        label3.pack()
        label4=Label(frame,text="It take approx 25 seconds to get your  diseases,so plz wait. Thankyou!",bg='light green')
        label4.pack(side=BOTTOM)
        frame.pack()
        button1=Button(root,text="syptom1",command=get_me1)
        button1.pack()
        button2=Button(root,text="syptom2",command=get_me2)
        button2.pack()
        button3=Button(root,text="syptom3",command=get_me3)
        button3.pack()
        button4=Button(root,text="syptom4",command=get_me4)
        button4.pack()
        button5=Button(root,text="syptom5",command=get_me5)
        button5.pack()
        button6=Button(root,text="syptom6",command=get_me6)
        button6.pack()
        button7=Button(root,text='exit',command=call_me)
        button7.pack()

        root.geometry("581x270")

        root.mainloop()


        #reading csv file

        df = pd.read_csv(r'dataset1.csv')

        i=df.iloc[:,1:]
        d=df.iloc[:,0:1]

        lb = LabelEncoder()
        d["Disease"]= lb.fit_transform(d["Disease"])

        nltk.download('stopwords')

        def text_cleaning(a):
         cleaning =[char for char in a if char not in string.punctuation]
         print(cleaning)
         cleaning=''.join(cleaning)
         print(cleaning)   
         return [word for word in cleaning.split() if word.lower() not in stopwords.words('english')]

        bow_transformer0 = CountVectorizer(analyzer=text_cleaning).fit(df['Symptom 0'])
        bow_transformer1 = CountVectorizer(analyzer=text_cleaning).fit(df['Symptom 1'])
        bow_transformer2 = CountVectorizer(analyzer=text_cleaning).fit(df['Symptom 2'])
        bow_transformer3 = CountVectorizer(analyzer=text_cleaning).fit(df['Symptom 3'])
        bow_transformer4 = CountVectorizer(analyzer=text_cleaning).fit(df['Symptom 4'])
        bow_transformer5 = CountVectorizer(analyzer=text_cleaning).fit(df['Symptom 5'])

        title_bow0 = bow_transformer0.transform(df['Symptom 0'])
        title_bow1 = bow_transformer1.transform(df['Symptom 1'])
        title_bow2 = bow_transformer2.transform(df['Symptom 2'])
        title_bow3 = bow_transformer3.transform(df['Symptom 3'])
        title_bow4 = bow_transformer4.transform(df['Symptom 4'])
        title_bow5 = bow_transformer5.transform(df['Symptom 5'])

        X0 = title_bow0.toarray()
        X1 = title_bow1.toarray()
        X2 = title_bow2.toarray()
        X3 = title_bow3.toarray()
        X4 = title_bow4.toarray()
        X5 = title_bow5.toarray()


        x_train0, x_test0, y_train0, y_test0 = train_test_split(X0, d,test_size=0.2, random_state=40)
        x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, d,test_size=0.2, random_state=40)
        x_train2, x_test2, y_train2, y_test2 = train_test_split(X2, d,test_size=0.2, random_state=40)
        x_train3, x_test3, y_train3, y_test3 = train_test_split(X3, d,test_size=0.2, random_state=40)
        x_train4, x_test4, y_train4, y_test4 = train_test_split(X4, d,test_size=0.2, random_state=40)
        x_train5, x_test5, y_train5, y_test5 = train_test_split(X5, d,test_size=0.2, random_state=40)


        clf=BernoulliNB()
        model0 = clf.fit(x_train0,y_train0)
        clf=BernoulliNB()
        model1 = clf.fit(x_train1,y_train1)
        clf=BernoulliNB()
        model2 = clf.fit(x_train2,y_train2)
        clf=BernoulliNB()
        model3 = clf.fit(x_train3,y_train3)
        clf=BernoulliNB()
        model4 = clf.fit(x_train4,y_train4)
        clf=BernoulliNB()
        model5 = clf.fit(x_train5,y_train5)

        unseen1= bow_transformer0.transform(ud1)
        unseen2= bow_transformer1.transform(ud2)
        unseen3= bow_transformer2.transform(ud3)
        unseen4= bow_transformer3.transform(ud4)
        unseen5= bow_transformer4.transform(ud5)
        unseen6= bow_transformer5.transform(ud6)

        a1=unseen1.toarray()
        a2=unseen2.toarray()
        a3=unseen3.toarray()
        a4=unseen4.toarray()
        a5=unseen5.toarray()
        a6=unseen6.toarray()

        print("According to your symptoms diseases are>>>>>>>>>>>>>>>>>>>>>>")

        print()

        print("for symptom1:-")

        predict1=model0.predict(a1)
        if(predict1==[0]):
          print("AIDS")
        elif(predict1==[1]):
          print("Acne")
        elif(predict1==[2]):
          print("Alcoholic hepatitis")
        elif(predict1==[3]):
          print("Allergy")
        elif(predict1==[4]):
          print("Arthritis")
        elif(predict1==[5]):
          print("Bronchial Asthma")
        elif(predict1==[6]):
          print("Cervical spondylosis")
        elif(predict1==[7]):
          print("Chicken pox")
        elif(predict1==[8]):
          print("Chronic cholestasis")  
        elif(predict1==[9]):
          print("Common Cold")
        elif(predict1==[10]):
          print("Covid")
        elif(predict1==[11]):
          print("Dengue")
        elif(predict1==[12]):
          print("Diabetes")
        elif(predict1==[13]):
          print("Dimorphic hemorrhoids(piles)")
        elif(predict1==[14]):
          print("Drug Reaction")
        elif(predict1==[15]):
          print("Fungal infection")
        elif(predict1==[16]):
          print("GRED")
        elif(predict1==[17]):
          print("Gastroenteritis")
        elif(predict1==[18]):
          print("Heart attack")
        elif(predict1==[19]):
          print("Hepatitis A")
        elif(predict1==[20]):
          print("Hepatitis B")
        elif(predict1==[21]):
          print("Hepatitis C")
        elif(predict1==[22]):
          print("Hepatitis D")
        elif(predict1==[23]):
          print("Hepatitis E")
        elif(predict1==[24]):
          print("Hypertension")
        elif(predict1==[25]):
          print("Hyperthyroidism")  
        elif(predict1==[26]):
          print("Hypoglycemia")
        elif(predict1==[27]):
          print("Hypothyroidism")
        elif(predict1==[28]):
          print("Impetigo")
        elif(predict1==[29]):
          print("Jaundice")
        elif(predict1==[30]):
          print("Malaria")
        elif(predict1==[31]):
          print("Migraine")
        elif(predict1==[32]):
          print("Osteoarthritis")
        elif(predict1==[33]):
          print("Paralysis (brain hemorrhage)")  
        elif(predict1==[34]):
          print("Paroxysmal Positional Vertigo")
        elif(predict1==[35]):
          print("Peptic ulcer disease")
        elif(predict1==[36]):
          print(" Pneumonia")
        elif(predict1==[37]):
          print("Psoriasis")
        elif(predict1==[38]):
          print("Tuberculosis")
        elif(predict1==[39]):
          print("Typhoid")
        elif(predict1==[40]):
          print("Urinary tract infection") 
        else:
          print("Disease not found sorry!")

        print()

        print("for symptom2:-")

        predict2=model1.predict(a2)
        if(predict2==[0]):
          print("AIDS")
        elif(predict2==[1]):
          print("Acne")
        elif(predict2==[2]):
          print("Alcoholic hepatitis")
        elif(predict2==[3]):
          print("Allergy")
        elif(predict2==[4]):
          print("Arthritis")
        elif(predict2==[5]):
          print("Bronchial Asthma")
        elif(predict2==[6]):
          print("Cervical spondylosis")
        elif(predict2==[7]):
          print("Chicken pox")
        elif(predict2==[8]):
          print("Chronic cholestasis")  
        elif(predict2==[9]):
          print("Common Cold")
        elif(predict2==[10]):
          print("Covid")
        elif(predict2==[11]):
          print("Dengue")
        elif(predict2==[12]):
          print("Diabetes")
        elif(predict2==[13]):
          print("Dimorphic hemorrhoids(piles)")
        elif(predict2==[14]):
          print("Drug Reaction")
        elif(predict2==[15]):
          print("Fungal infection")
        elif(predict2==[16]):
          print("GRED")
        elif(predict2==[17]):
          print("Gastroenteritis")
        elif(predict2==[18]):
          print("Heart attack")
        elif(predict2==[19]):
          print("Hepatitis A")
        elif(predict2==[20]):
          print("Hepatitis B")
        elif(predict2==[21]):
          print("Hepatitis C")
        elif(predict2==[22]):
          print("Hepatitis D")
        elif(predict2==[23]):
          print("Hepatitis E")
        elif(predict2==[24]):
          print("Hypertension")
        elif(predict2==[25]):
          print("Hyperthyroidism")  
        elif(predict2==[26]):
          print("Hypoglycemia")
        elif(predict2==[27]):
          print("Hypothyroidism")
        elif(predict2==[28]):
          print("Impetigo")
        elif(predict2==[29]):
          print("Jaundice")
        elif(predict2==[30]):
          print("Malaria")
        elif(predict2==[31]):
          print("Migraine")
        elif(predict2==[32]):
          print("Osteoarthritis")
        elif(predict2==[33]):
          print("Paralysis (brain hemorrhage)")  
        elif(predict2==[34]):
          print("Paroxysmal Positional Vertigo")
        elif(predict2==[35]):
          print("Peptic ulcer disease")
        elif(predict2==[36]):
          print(" Pneumonia")
        elif(predict2==[37]):
          print("Psoriasis")
        elif(predict2==[38]):
          print("Tuberculosis")
        elif(predict2==[39]):
          print("Typhoid")
        elif(predict2==[40]):
          print("Urinary tract infection") 
        else:
          print("Disease not found sorry!")

        print()

        print("for symptom3:-")

        predict3=model2.predict(a3)
        if(predict3==[0]):
          print("AIDS")
        elif(predict3==[1]):
          print("Acne")
        elif(predict3==[2]):
          print("Alcoholic hepatitis")
        elif(predict3==[3]):
          print("Allergy")
        elif(predict3==[4]):
          print("Arthritis")
        elif(predict3==[5]):
          print("Bronchial Asthma")
        elif(predict3==[6]):
          print("Cervical spondylosis")
        elif(predict3==[7]):
          print("Chicken pox")
        elif(predict3==[8]):
          print("Chronic cholestasis")  
        elif(predict3==[9]):
          print("Common Cold")
        elif(predict3==[10]):
          print("Covid")
        elif(predict3==[11]):
          print("Dengue")
        elif(predict3==[12]):
          print("Diabetes")
        elif(predict3==[13]):
          print("Dimorphic hemorrhoids(piles)")
        elif(predict3==[14]):
          print("Drug Reaction")
        elif(predict3==[15]):
          print("Fungal infection")
        elif(predict3==[16]):
          print("GRED")
        elif(predict3==[17]):
          print("Gastroenteritis")
        elif(predict3==[18]):
          print("Heart attack")
        elif(predict3==[19]):
          print("Hepatitis A")
        elif(predict3==[20]):
          print("Hepatitis B")
        elif(predict3==[21]):
          print("Hepatitis C")
        elif(predict3==[22]):
          print("Hepatitis D")
        elif(predict3==[23]):
          print("Hepatitis E")
        elif(predict3==[24]):
          print("Hypertension")
        elif(predict3==[25]):
          print("Hyperthyroidism")  
        elif(predict3==[26]):
          print("Hypoglycemia")
        elif(predict3==[27]):
          print("Hypothyroidism")
        elif(predict3==[28]):
          print("Impetigo")
        elif(predict3==[29]):
          print("Jaundice")
        elif(predict3==[30]):
          print("Malaria")
        elif(predict3==[31]):
          print("Migraine")
        elif(predict3==[32]):
          print("Osteoarthritis")
        elif(predict3==[33]):
          print("Paralysis (brain hemorrhage)")  
        elif(predict3==[34]):
          print("Paroxysmal Positional Vertigo")
        elif(predict3==[35]):
          print("Peptic ulcer disease")
        elif(predict3==[36]):
          print(" Pneumonia")
        elif(predict3==[37]):
          print("Psoriasis")
        elif(predict3==[38]):
          print("Tuberculosis")
        elif(predict3==[39]):
          print("Typhoid")
        elif(predict3==[40]):
          print("Urinary tract infection") 
        else:
          print("Disease not found sorry!")

        print()

        print("for symptom4:-")

        predict4=model3.predict(a4)
        if(predict4==[0]):
          print("AIDS")
        elif(predict4==[1]):
          print("Acne")
        elif(predict4==[2]):
          print("Alcoholic hepatitis")
        elif(predict4==[3]):
          print("Allergy")
        elif(predict4==[4]):
          print("Arthritis")
        elif(predict4==[5]):
          print("Bronchial Asthma")
        elif(predict4==[6]):
          print("Cervical spondylosis")
        elif(predict4==[7]):
          print("Chicken pox")
        elif(predict4==[8]):
          print("Chronic cholestasis")  
        elif(predict4==[9]):
          print("Common Cold")
        elif(predict4==[10]):
          print("Covid")
        elif(predict4==[11]):
          print("Dengue")
        elif(predict4==[12]):
          print("Diabetes")
        elif(predict4==[13]):
          print("Dimorphic hemorrhoids(piles)")
        elif(predict4==[14]):
          print("Drug Reaction")
        elif(predict4==[15]):
          print("Fungal infection")
        elif(predict4==[16]):
          print("GRED")
        elif(predict4==[17]):
          print("Gastroenteritis")
        elif(predict4==[18]):
          print("Heart attack")
        elif(predict4==[19]):
          print("Hepatitis A")
        elif(predict4==[20]):
          print("Hepatitis B")
        elif(predict4==[21]):
          print("Hepatitis C")
        elif(predict4==[22]):
          print("Hepatitis D")
        elif(predict4==[23]):
          print("Hepatitis E")
        elif(predict4==[24]):
          print("Hypertension")
        elif(predict4==[25]):
          print("Hyperthyroidism")  
        elif(predict4==[26]):
          print("Hypoglycemia")
        elif(predict4==[27]):
          print("Hypothyroidism")
        elif(predict4==[28]):
          print("Impetigo")
        elif(predict4==[29]):
          print("Jaundice")
        elif(predict4==[30]):
          print("Malaria")
        elif(predict4==[31]):
          print("Migraine")
        elif(predict4==[32]):
          print("Osteoarthritis")
        elif(predict4==[33]):
          print("Paralysis (brain hemorrhage)")  
        elif(predict4==[34]):
          print("Paroxysmal Positional Vertigo")
        elif(predict4==[35]):
          print("Peptic ulcer disease")
        elif(predict4==[36]):
          print(" Pneumonia")
        elif(predict4==[37]):
          print("Psoriasis")
        elif(predict4==[38]):
          print("Tuberculosis")
        elif(predict4==[39]):
          print("Typhoid")
        elif(predict4==[40]):
          print("Urinary tract infection") 
        else:
          print("Disease not found sorry!")

        print()

        print("for symptom5:-")

        predict5=model4.predict(a5)
        if(predict5==[0]):
          print("AIDS")
        elif(predict5==[1]):
          print("Acne")
        elif(predict5==[2]):
          print("Alcoholic hepatitis")
        elif(predict5==[3]):
          print("Allergy")
        elif(predict5==[4]):
          print("Arthritis")
        elif(predict5==[5]):
          print("Bronchial Asthma")
        elif(predict5==[6]):
          print("Cervical spondylosis")
        elif(predict5==[7]):
          print("Chicken pox")
        elif(predict5==[8]):
          print("Chronic cholestasis")  
        elif(predict5==[9]):
          print("Common Cold")
        elif(predict5==[10]):
          print("Covid")
        elif(predict5==[11]):
          print("Dengue")
        elif(predict5==[12]):
          print("Diabetes")
        elif(predict5==[13]):
          print("Dimorphic hemorrhoids(piles)")
        elif(predict5==[14]):
          print("Drug Reaction")
        elif(predict5==[15]):
          print("Fungal infection")
        elif(predict5==[16]):
          print("GRED")
        elif(predict5==[17]):
          print("Gastroenteritis")
        elif(predict5==[18]):
          print("Heart attack")
        elif(predict5==[19]):
          print("Hepatitis A")
        elif(predict5==[20]):
          print("Hepatitis B")
        elif(predict5==[21]):
          print("Hepatitis C")
        elif(predict5==[22]):
          print("Hepatitis D")
        elif(predict5==[23]):
          print("Hepatitis E")
        elif(predict5==[24]):
          print("Hypertension")
        elif(predict5==[25]):
          print("Hyperthyroidism")  
        elif(predict5==[26]):
          print("Hypoglycemia")
        elif(predict5==[27]):
          print("Hypothyroidism")
        elif(predict5==[28]):
          print("Impetigo")
        elif(predict5==[29]):
          print("Jaundice")
        elif(predict5==[30]):
          print("Malaria")
        elif(predict5==[31]):
          print("Migraine")
        elif(predict5==[32]):
          print("Osteoarthritis")
        elif(predict5==[33]):
          print("Paralysis (brain hemorrhage)")  
        elif(predict5==[34]):
          print("Paroxysmal Positional Vertigo")
        elif(predict5==[35]):
          print("Peptic ulcer disease")
        elif(predict5==[36]):
          print(" Pneumonia")
        elif(predict5==[37]):
          print("Psoriasis")
        elif(predict5==[38]):
          print("Tuberculosis")
        elif(predict5==[39]):
          print("Typhoid")
        elif(predict5==[40]):
          print("Urinary tract infection") 
        else:
          print("Disease not found sorry!")

        print()

        print("for symptom6:-")

        predict6=model5.predict(a6)
        if(predict6==[0]):
          print("AIDS")
        elif(predict6==[1]):
          print("Acne")
        elif(predict6==[2]):
          print("Alcoholic hepatitis")
        elif(predict6==[3]):
          print("Allergy")
        elif(predict6==[4]):
          print("Arthritis")
        elif(predict6==[5]):
          print("Bronchial Asthma")
        elif(predict6==[6]):
          print("Cervical spondylosis")
        elif(predict6==[7]):
          print("Chicken pox")
        elif(predict6==[8]):
          print("Chronic cholestasis")  
        elif(predict6==[9]):
          print("Common Cold")
        elif(predict6==[10]):
          print("Covid")
        elif(predict6==[11]):
          print("Dengue")
        elif(predict6==[12]):
          print("Diabetes")
        elif(predict6==[13]):
          print("Dimorphic hemorrhoids(piles)")
        elif(predict6==[14]):
          print("Drug Reaction")
        elif(predict6==[15]):
          print("Fungal infection")
        elif(predict6==[16]):
          print("GRED")
        elif(predict6==[17]):
          print("Gastroenteritis")
        elif(predict6==[18]):
          print("Heart attack")
        elif(predict6==[19]):
          print("Hepatitis A")
        elif(predict6==[20]):
          print("Hepatitis B")
        elif(predict6==[21]):
          print("Hepatitis C")
        elif(predict6==[22]):
          print("Hepatitis D")
        elif(predict6==[23]):
          print("Hepatitis E")
        elif(predict6==[24]):
          print("Hypertension")
        elif(predict6==[25]):
          print("Hyperthyroidism")  
        elif(predict6==[26]):
          print("Hypoglycemia")
        elif(predict6==[27]):
          print("Hypothyroidism")
        elif(predict6==[28]):
          print("Impetigo")
        elif(predict6==[29]):
          print("Jaundice")
        elif(predict6==[30]):
          print("Malaria")
        elif(predict6==[31]):
          print("Migraine")
        elif(predict6==[32]):
          print("Osteoarthritis")
        elif(predict6==[33]):
          print("Paralysis (brain hemorrhage)")  
        elif(predict6==[34]):
          print("Paroxysmal Positional Vertigo")
        elif(predict6==[35]):
          print("Peptic ulcer disease")
        elif(predict6==[36]):
          print(" Pneumonia")
        elif(predict6==[37]):
          print("Psoriasis")
        elif(predict6==[38]):
          print("Tuberculosis")
        elif(predict6==[39]):
          print("Typhoid")
        elif(predict6==[40]):
          print("Urinary tract infection") 
        else:
          print("Disease not found sorry!")
          
    elif(ud7=='n'):
        print()
        print("thank you for visiting!")
        sys.exit()
    else:
        print("invalid input. Try again!")

    ud7=input("Do you want to continue?(y/n): ")



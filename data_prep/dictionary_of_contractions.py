import re

CONTRACTIONS={
	re.compile(r"\bcannot\b"): "can not",
	re.compile(r"\bisn ?t\b"):"is not",
	re.compile(r"\baren ?t\b"):"are not",
	re.compile(r"\bwasn ?t\b"):"was not",
	re.compile(r"\bweren ?t\b"):"were not",
	re.compile(r"\bhaven ?t\b"):"have not",
	re.compile(r"\bhasn ?t\b"):"has not",
	re.compile(r"\bhadn ?t\b"):"had not",
	re.compile(r"\bwon ?t\b"):"will not",
	re.compile(r"\bwouldn ?t\b"):"would not",
	re.compile(r"\bdon ?t\b"):"do not",
	re.compile(r"\bdoesn ?t\b"):"does not",
	re.compile(r"\bdidn ?t\b"):"did not",
	re.compile(r"\bcan ?t\b"):"can not",
	re.compile(r"\bcouldn ?t\b"):"could not",
	re.compile(r"\bshouldn ?t\b"):"should not",
	re.compile(r"\bmightn ?t\b"):"might not",
	re.compile(r"\bmustn ?t\b"):"must not",
	re.compile(r"\bi ?m\b"):"i am",
	re.compile(r"\bi ?ll\b"):"i will",
	re.compile(r"\bi ?d\b"):"i had", #had/would
	re.compile(r"\bi ?ve\b"):"i have",
	re.compile(r"\byou re\b"):"you are",
	re.compile(r"\byou ?ll\b"):"you will",
	re.compile(r"\byou ?d\b"):"you had", #had/would
	re.compile(r"\byou ?ve\b"):"you have",
	re.compile(r"\bhe ?s\b"):"he is",
	re.compile(r"\bhe ?ll\b"):"he will",
	re.compile(r"\bhe ?d\b"):"he had",  #had/would
	re.compile(r"\bhe ?s\b"):"he has",
	re.compile(r"\bshe ?s\b"):"she is",
	re.compile(r"\bshe ?ll\b"):"she will",
	re.compile(r"\bshe ?d\b"):"she had",  #had/would
	re.compile(r"\bshe ?s\b"):"she has",
	re.compile(r"\bit ?s\b"):"it is",
	re.compile(r"\bit ?ll\b"):"it will",
	re.compile(r"\bit ?d\b"):"it had",  #had/would
	re.compile(r"\bit ?s\b"):"it has",
	re.compile(r"\bwe re\b"):"we are",
	re.compile(r"\bwe ?ll\b"):"we will",
	re.compile(r"\bwe ?d\b"):"we had",  #had/would
	re.compile(r"\bwe ?ve\b"):"we have",
	re.compile(r"\bthey re\b"):"they are",
	re.compile(r"\bthey ?ll\b"):"they will",
	re.compile(r"\bthey ?d\b"):"they had",  #had/would
	re.compile(r"\bthey ?ve\b"):"they have",
	re.compile(r"\bthat ?s\b"):"that is",
	re.compile(r"\bthat ?ll\b"):"that will",
	re.compile(r"\bthat ?d\b"):"that had",  #had/would
	re.compile(r"\bthat ?s\b"):"that has",
	re.compile(r"\bwho ?s\b"):"who is",
	re.compile(r"\bwho ?ll\b"):"who will",
	re.compile(r"\bwho ?d\b"):"who had",  #had/would
	re.compile(r"\bwho ?s\b"):"who has",
	re.compile(r"\bwhat ?s\b"):"what is",
	re.compile(r"\bwhat re\b"):"what are",
	re.compile(r"\bwhat ?ll\b"):"what will",
	re.compile(r"\bwhat ?d\b"):"what had",  #had/would
	re.compile(r"\bwhat ?s\b"):"what has",
	re.compile(r"\bwhere ?s\b"):"where is",
	re.compile(r"\bwhere ?ll\b"):"where will",
	re.compile(r"\bwhere ?d\b"):"where had", #had/would
	re.compile(r"\bwhere ?s\b"):"where has",
	re.compile(r"\bwhen ?s\b"):"when is",
	re.compile(r"\bwhen ?ll\b"):"when will",
	re.compile(r"\bwhen ?d\b"):"when had", #had/would
	re.compile(r"\bwhen ?s\b"):"when has",
	re.compile(r"\bwhy ?s\b"):"why is",
	re.compile(r"\bwhy ?ll\b"):"why will",
	re.compile(r"\bwhy ?d\b"):"why had", #had/would
	re.compile(r"\bwhy ?s\b"):"why has",
	re.compile(r"\bhow ?s\b"):"how is",
	re.compile(r"\bhow ?ll\b"):"how will",
	re.compile(r"\bhow ?d\b"):"how had",  #had/would
	re.compile(r"\bhow ?s\b"):"how has",
	re.compile(r"\bya ?ll\b"):"you all"
}


def contractions(tweet):
    for contraction in CONTRACTIONS:		#Remove contractions
        tweet=contraction.sub(CONTRACTIONS[contraction],tweet)
    return tweet



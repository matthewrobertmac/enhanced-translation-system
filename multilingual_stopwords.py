"""
Multilingual stopword dictionary for wordcloud filtering.
Includes pronouns, articles, prepositions, conjunctions,
auxiliary verbs, and filler words for all supported languages.
"""

MULTILINGUAL_STOPWORDS = {

    # ----------------------------------------------------------
    # ENGLISH
    # ----------------------------------------------------------
    "English (US)": {
        "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
        "yourself","yourselves","he","him","his","himself","she","her","hers",
        "herself","it","its","itself","they","them","their","theirs","themselves",
        "what","which","who","whom","this","that","these","those","a","an","the",
        "and","but","if","or","because","as","until","while","of","at","by","for",
        "with","about","against","between","into","through","during","before",
        "after","above","below","to","from","up","down","in","out","on","off",
        "over","under","again","further","then","once","here","there","when",
        "where","why","how","all","any","both","each","few","more","most","other",
        "some","such","no","nor","not","only","own","same","so","than","too","very",

        # Missing prepositions & connectors
        "onto","upon","via","per","throughout","towards","within","without",
        "across","amid","among","beside","beyond","despite","except","inside",
        "outside"
    },

    "English (UK)": set(),  # mapped to English (US)

    # ----------------------------------------------------------
    # SPANISH
    # ----------------------------------------------------------
    "Spanish": {
        "yo","me","mí","conmigo","tú","te","ti","él","ella","nosotros","nosotras",
        "vosotros","vosotras","usted","ustedes","mi","mis","tu","tus","su","sus",
        "nuestro","nuestra","nuestros","nuestras","lo","la","los","las","nos","os",
        "le","les","se","a","ante","bajo","cabe","con","contra","de","desde","en",
        "entre","hacia","hasta","para","por","según","sin","sobre","tras","el","la",
        "los","las","un","una","unos","unas","y","o","pero",

        # Added connectors & missed prepositions
        "mediante","durante","excepto","salvo","incluso","además","aunque",
        "mientras","cuando","donde","como","segun","ya","aun","aún","pues","entonces"
    },

    # ----------------------------------------------------------
    # FRENCH
    # ----------------------------------------------------------
    "French": {
        "je","tu","il","elle","nous","vous","ils","elles","me","te","se","le","la",
        "les","un","une","des","du","de","en","à","au","aux","dans","sur","sous",
        "chez","par","pour","avec","sans","mais","ou","et","ce","cet","cette","ces",

        # Added connectors
        "vers","depuis","pendant","lorsque","puis","car","donc","tandis","ainsi",
        "comme","quoi","cela","ceci"
    },

    # ----------------------------------------------------------
    # GERMAN
    # ----------------------------------------------------------
    "German": {
        "ich","du","er","sie","es","wir","ihr","sie","mich","dich","uns","euch",
        "mein","dein","sein","ihr","unser","euer","der","die","das","ein","eine",
        "und","oder","aber","denn","sondern","in","im","am","aus","mit","ohne",
        "zu","vom","beim","für",

        # Added prepositions
        "über","unter","während","seit","wegen","trotz","bevor","nachdem",
        "weil","obwohl","damit","daher"
    },

    # ----------------------------------------------------------
    # ITALIAN
    # ----------------------------------------------------------
    "Italian": {
        "io","tu","lui","lei","noi","voi","loro","mi","ti","ci","vi","si","il",
        "lo","la","i","gli","le","un","una","uno","di","a","da","in","con","su",
        "per","tra","fra","e","o","ma",

        # Added connectors
        "verso","oltre","sotto","sopra","durante","mentre","poiché","perché",
        "dunque","quindi","come","dove","quando","anche"
    },

    # ----------------------------------------------------------
    # PORTUGUESE
    # ----------------------------------------------------------
    "Portuguese": {
        "eu","tu","ele","ela","nós","vós","eles","elas","me","te","se","nos",
        "vos","o","a","os","as","um","uma","uns","umas","de","em","para","por",
        "com","sem","sobre","entre","ao","à","às",

        # Added connectors
        "desde","contra","durante","através","embora","porque","que","como",
        "quando","onde","também"
    },

    # ----------------------------------------------------------
    # ROMANIAN
    # ----------------------------------------------------------
    "Romanian": {
        "eu","tu","el","ea","noi","voi","ei","ele","me","te","îl","o","ne",
        "vă","îi","le","și","sau","dar","că","în","pe","la","din","cu","fără",
        "pentru",

        # Added connectors
        "între","până","deoarece","căci","astfel","însă","totuși","acum","aici",
        "aceasta","acela"
    },

    # ----------------------------------------------------------
    # POLISH
    # ----------------------------------------------------------
    "Polish": {
        "ja","ty","on","ona","ono","my","wy","oni","one","mnie","tobie","jego",
        "jej","nam","wam","ich","i","lub","ale","albo","to","że","do","na","w",
        "z","bez","przez",

        # Added prepositions
        "pod","przy","nad","za","od","gdy","kiedy","ponieważ","więc","także"
    },

    # ----------------------------------------------------------
    # UKRAINIAN
    # ----------------------------------------------------------
    "Ukrainian": {
        "я","ти","він","вона","воно","ми","ви","вони","мені","тебе","його","її",
        "нас","вас","їх","і","та","але","або","до","у","в","на","з","із","без",
        "при",

        # Added connectors
        "під","над","перед","через","поки","коли","тому","який","яка","яке","які"
    },

    # ----------------------------------------------------------
    # RUSSIAN
    # ----------------------------------------------------------
    "Russian": {
        "я","ты","он","она","оно","мы","вы","они","меня","тебя","его","ее","нас",
        "вас","их","и","но","или","а","же","да","нет","в","на","по","из","без",
        "для","о","об",

        # Added connectors
        "под","над","через","перед","пока","когда","потому","который","которая",
        "которое","которые"
    },

    # ----------------------------------------------------------
    # CZECH
    # ----------------------------------------------------------
    "Czech": {
        "já","ty","on","ona","ono","my","vy","oni","mě","tě","ho","ji","nás",
        "vás","je","a","nebo","ale","do","na","v","s","bez","pro","o","u",

        # Added prepositions
        "pod","nad","před","za","protože","když","který","která","které"
    },

    # ----------------------------------------------------------
    # SLOVAK
    # ----------------------------------------------------------
    "Slovak": {
        "ja","ty","on","ona","ono","my","vy","oni","ony","mňa","teba","jeho",
        "jej","nás","vás","ich","a","ale","alebo","do","na","v","s","bez","pre",

        # Added connectors
        "pod","nad","pred","za","pretože","keď","ktorý","ktorá","ktoré"
    },

    # ----------------------------------------------------------
    # BULGARIAN
    # ----------------------------------------------------------
    "Bulgarian": {
        "аз","ти","той","тя","то","ние","вие","те","мен","теб","него","нея",
        "нас","вас","тях","и","или","но","че","в","на","с","от","за","до","без",

        # Added connectors
        "преди","след","над","под","когато","докато","защо","който","която",
        "което","които"
    },

    # ----------------------------------------------------------
    # DUTCH
    # ----------------------------------------------------------
    "Dutch": {
        "ik","jij","hij","zij","wij","jullie","zij","mij","je","hem","haar",
        "ons","hun","de","het","een","en","of","maar","want","dus","bij","in",
        "op","aan","met","voor",

        # Added connectors
        "onder","boven","achter","voorbij","tijdens","sinds","waarom","waar",
        "wanneer","omdat","zoals","toch","ook"
    },
}
# ----------------------------------------------------------
# LANGUAGE ALIASES FOR ENGLISH — unify all variations
# ----------------------------------------------------------
MULTILINGUAL_STOPWORDS["English"] = MULTILINGUAL_STOPWORDS["English (US)"]
MULTILINGUAL_STOPWORDS["english"] = MULTILINGUAL_STOPWORDS["English (US)"]
MULTILINGUAL_STOPWORDS["EN"] = MULTILINGUAL_STOPWORDS["English (US)"]
MULTILINGUAL_STOPWORDS["en"] = MULTILINGUAL_STOPWORDS["English (US)"]
MULTILINGUAL_STOPWORDS["en-US"] = MULTILINGUAL_STOPWORDS["English (US)"]
MULTILINGUAL_STOPWORDS["English US"] = MULTILINGUAL_STOPWORDS["English (US)"]
MULTILINGUAL_STOPWORDS["English (UK)"] = MULTILINGUAL_STOPWORDS["English (US)"]

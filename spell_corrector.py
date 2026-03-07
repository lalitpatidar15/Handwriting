from rapidfuzz import process, fuzz
import re

class GeneralSpellCorrector:
    """
    General-purpose spelling corrector for handwritten text.
    Uses common English words dictionary and fuzzy matching.
    """
    
    def __init__(self):
        # Common English words dictionary (most frequently used words)
        # Focus on words commonly found in handwritten notes
        self.common_words = [
            # Basic words
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their",
            "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
            "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could", "them",
            "see", "other", "than", "then", "now", "look", "only", "come", "its", "over",
            "think", "also", "back", "after", "use", "two", "how", "our", "work", "first",
            "well", "way", "even", "new", "want", "because", "any", "these", "give", "day",
            "most", "us", "is", "are", "was", "were", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "should", "could", "may", "might", "must",
            "can", "cannot", "can't", "don't", "doesn't", "didn't", "won't", "wouldn't",
            "shouldn't", "couldn't", "mustn't", "isn't", "aren't", "wasn't", "weren't",
            "haven't", "hasn't", "hadn't", "let's", "that's", "it's", "he's", "she's",
            "you're", "we're", "they're", "i'm", "here's", "there's", "what's", "who's",
            # Common verbs
            "tell", "told", "telling", "tells", "says", "said", "saying",
            "get", "got", "getting", "gets", "gotten",
            "go", "went", "going", "goes", "gone",
            "come", "came", "coming", "comes",
            "know", "knew", "knowing", "knows", "known",
            "think", "thought", "thinking", "thinks",
            "see", "saw", "seeing", "sees", "seen",
            "want", "wanted", "wanting", "wants",
            "need", "needed", "needing", "needs",
            "try", "tried", "trying", "tries",
            "help", "helped", "helping", "helps",
            "work", "worked", "working", "works",
            "play", "played", "playing", "plays",
            "run", "ran", "running", "runs",
            "walk", "walked", "walking", "walks",
            "talk", "talked", "talking", "talks",
            "write", "wrote", "writing", "writes", "written",
            "read", "reading", "reads",
            "give", "gave", "giving", "gives", "given",
            "take", "took", "taking", "takes", "taken",
            "make", "made", "making", "makes",
            "find", "found", "finding", "finds",
            "keep", "kept", "keeping", "keeps",
            "let", "letting", "lets",
            "put", "putting", "puts",
            "set", "setting", "sets",
            "say", "saying", "says",
            "ask", "asked", "asking", "asks",
            "show", "showed", "showing", "shows", "shown",
            "move", "moved", "moving", "moves",
            "live", "lived", "living", "lives",
            "believe", "believed", "believing", "believes",
            "bring", "brought", "bringing", "brings",
            "happen", "happened", "happening", "happens",
            "provide", "provided", "providing", "provides",
            "sit", "sat", "sitting", "sits",
            "stand", "stood", "standing", "stands",
            "lose", "lost", "losing", "loses",
            "add", "added", "adding", "adds",
            "change", "changed", "changing", "changes",
            "die", "died", "dying", "dies",
            "cut", "cutting", "cuts",
            "eat", "ate", "eating", "eats", "eaten",
            "fall", "fell", "falling", "falls", "fallen",
            "feel", "felt", "feeling", "feels",
            "fight", "fought", "fighting", "fights",
            "fill", "filled", "filling", "fills",
            "hit", "hitting", "hits",
            "hold", "held", "holding", "holds",
            "hurt", "hurting", "hurts",
            "kill", "killed", "killing", "kills",
            "kiss", "kissed", "kissing", "kisses",
            "laugh", "laughed", "laughing", "laughs",
            "lay", "laid", "laying", "lays",
            "lead", "led", "leading", "leads",
            "leave", "left", "leaving", "leaves",
            "lie", "lay", "lying", "lies", "lain",
            "light", "lit", "lighting", "lights",
            "like", "liked", "liking", "likes",
            "listen", "listened", "listening", "listens",
            "look", "looked", "looking", "looks",
            "love", "loved", "loving", "loves",
            "meet", "met", "meeting", "meets",
            "pay", "paid", "paying", "pays",
            "read", "reading", "reads",
            "ride", "rode", "riding", "rides", "ridden",
            "ring", "rang", "ringing", "rings", "rung",
            "rise", "rose", "rising", "rises", "risen",
            "sell", "sold", "selling", "sells",
            "send", "sent", "sending", "sends",
            "shake", "shook", "shaking", "shakes", "shaken",
            "shine", "shone", "shining", "shines",
            "shoot", "shot", "shooting", "shoots",
            "shut", "shutting", "shuts",
            "sing", "sang", "singing", "sings", "sung",
            "sink", "sank", "sinking", "sinks", "sunk",
            "sleep", "slept", "sleeping", "sleeps",
            "smell", "smelled", "smelling", "smells",
            "speak", "spoke", "speaking", "speaks", "spoken",
            "spend", "spent", "spending", "spends",
            "steal", "stole", "stealing", "steals", "stolen",
            "stick", "stuck", "sticking", "sticks",
            "strike", "struck", "striking", "strikes", "struck",
            "swim", "swam", "swimming", "swims", "swum",
            "swing", "swung", "swinging", "swings",
            "teach", "taught", "teaching", "teaches",
            "tear", "tore", "tearing", "tears", "torn",
            "throw", "threw", "throwing", "throws", "thrown",
            "understand", "understood", "understanding", "understands",
            "wake", "woke", "waking", "wakes", "woken",
            "wear", "wore", "wearing", "wears", "worn",
            "win", "won", "winning", "wins",
            # Common nouns
            "dream", "dreams", "people", "person", "persons",
            "thing", "things", "way", "ways", "day", "days",
            "man", "men", "woman", "women", "child", "children",
            "time", "times", "year", "years", "week", "weeks",
            "month", "months", "hour", "hours", "minute", "minutes",
            "place", "places", "point", "points", "home", "homes",
            "hand", "hands", "part", "parts", "life", "lives",
            "world", "worlds", "school", "schools", "house", "houses",
            "room", "rooms", "door", "doors", "window", "windows",
            "car", "cars", "book", "books", "word", "words",
            "number", "numbers", "name", "names", "friend", "friends",
            "family", "families", "mother", "mothers", "father", "fathers",
            "brother", "brothers", "sister", "sisters", "son", "sons",
            "daughter", "daughters", "baby", "babies", "boy", "boys",
            "girl", "girls", "student", "students", "teacher", "teachers",
            "doctor", "doctors", "nurse", "nurses", "patient", "patients",
            "food", "foods", "water", "waters", "money", "moneys",
            "job", "jobs", "problem", "problems",
            "question", "questions", "answer", "answers", "idea", "ideas",
            "story", "stories", "news", "game", "games",
            "music", "movie", "movies", "picture", "pictures",
            "phone", "phones", "computer", "computers", "internet",
            "email", "emails", "message", "messages", "letter", "letters",
            "paper", "papers", "pen", "pens", "pencil", "pencils",
            # Common phrases
            "something", "someone", "somewhere", "sometime", "sometimes", "somehow",
            "anything", "anyone", "anywhere", "anytime", "anyhow", "anyway",
            "everything", "everyone", "everywhere", "everytime", "everyday",
            "nothing", "noone", "nowhere", "nobody",
            # Common adjectives
            "good", "better", "best", "bad", "worse", "worst",
            "new", "newer", "newest", "old", "older", "oldest",
            "young", "younger", "youngest", "hot", "hotter", "hottest",
            "cold", "colder", "coldest", "warm", "warmer", "warmest",
            "cool", "cooler", "coolest", "nice", "nicer", "nicest",
            "beautiful", "pretty", "prettier", "prettiest", "ugly", "uglier", "ugliest",
            "happy", "happier", "happiest", "sad", "sadder", "saddest",
            "angry", "angrier", "angriest", "calm", "calmer", "calmest",
            "excited", "tired", "sleepy", "sleepier", "sleepiest",
            "hungry", "hungrier", "hungriest", "thirsty", "thirstier", "thirstiest",
            "full", "fuller", "fullest", "empty", "emptier", "emptiest",
            "clean", "cleaner", "cleanest", "dirty", "dirtier", "dirtiest",
            "easy", "easier", "easiest", "hard", "harder", "hardest",
            "difficult", "simple", "simpler", "simplest", "complex",
            "important", "interesting", "boring", "fun", "funny", "funnier", "funniest",
            "serious", "quiet", "quieter", "quietest", "loud", "louder", "loudest",
            "fast", "faster", "fastest", "slow", "slower", "slowest",
            "quick", "quicker", "quickest", "late", "later", "latest",
            "early", "earlier", "earliest", "soon", "sooner", "soonest",
            # Common phrases from the example
            "pursuit", "pursuits", "happiness", "protect", "protects", "protected", "protecting",
            "period", "periods", "right", "rights",
        ]
        
        # Common misspellings dictionary
        self.common_misspellings = {
            "teh": "the", "adn": "and", "taht": "that", "hte": "the",
            "yuo": "you", "yuor": "your", "recieve": "receive",
            "seperate": "separate", "occured": "occurred",
            "begining": "beginning", "enviroment": "environment",
            "definately": "definitely", "neccessary": "necessary",
            "accomodate": "accommodate", "embarass": "embarrass",
            "existance": "existence", "occassion": "occasion",
            "seige": "siege", "thier": "their", "wierd": "weird",
            "acheive": "achieve", "beleive": "believe",
            "calender": "calendar", "cemetery": "cemetery",
            "definite": "definite", "desperate": "desperate",
            "disappear": "disappear", "disappoint": "disappoint",
            "ecstasy": "ecstasy", "embarrass": "embarrass",
            "environment": "environment", "existence": "existence",
            "fascinate": "fascinate", "grateful": "grateful",
            "guarantee": "guarantee", "harass": "harass",
            "height": "height", "hierarchy": "hierarchy",
            "humorous": "humorous", "ignorance": "ignorance",
            "immediate": "immediate", "independent": "independent",
            "indispensable": "indispensable", "inoculate": "inoculate",
            "intelligence": "intelligence", "jewelry": "jewelry",
            "judgment": "judgment", "leisure": "leisure",
            "liaison": "liaison", "library": "library",
            "license": "license", "maintenance": "maintenance",
            "maneuver": "maneuver", "medieval": "medieval",
            "memento": "memento", "millennium": "millennium",
            "miniature": "miniature", "minuscule": "minuscule",
            "mischievous": "mischievous", "misspell": "misspell",
            "mortgage": "mortgage", "naive": "naive",
            "necessary": "necessary", "neighbor": "neighbor",
            "noticeable": "noticeable", "occasion": "occasion",
            "occurrence": "occurrence", "pamphlet": "pamphlet",
            "parallel": "parallel", "pastime": "pastime",
            "perseverance": "perseverance", "personnel": "personnel",
            "playwright": "playwright", "possession": "possession",
            "precede": "precede", "prejudice": "prejudice",
            "principal": "principal", "principle": "principle",
            "privilege": "privilege", "proceed": "proceed",
            "pronunciation": "pronunciation", "psychology": "psychology",
            "publicly": "publicly", "questionnaire": "questionnaire",
            "receipt": "receipt", "receive": "receive",
            "recommend": "recommend", "referring": "referring",
            "relevant": "relevant", "religious": "religious",
            "remember": "remember", "remembrance": "remembrance",
            "restaurant": "restaurant", "rhyme": "rhyme",
            "rhythm": "rhythm", "sacrilegious": "sacrilegious",
            "schedule": "schedule", "secretary": "secretary",
            "seize": "seize", "separate": "separate",
            "sergeant": "sergeant", "severely": "severely",
            "sincerely": "sincerely", "sophomore": "sophomore",
            "sponsor": "sponsor", "subtle": "subtle",
            "supersede": "supersede", "surprise": "surprise",
            "thorough": "thorough", "through": "through",
            "tolerance": "tolerance", "tomorrow": "tomorrow",
            "tongue": "tongue", "truly": "truly",
            "twelfth": "twelfth", "tyranny": "tyranny",
            "until": "until", "vacuum": "vacuum",
            "vegetable": "vegetable", "vehicle": "vehicle",
            "vicious": "vicious", "village": "village",
            "villain": "villain", "Wednesday": "Wednesday",
            "weird": "weird", "whether": "whether",
            "which": "which", "wholly": "wholly",
            "whose": "whose", "writing": "writing",
            "written": "written", "yacht": "yacht",
            "yield": "yield", "yolk": "yolk",
            "you're": "you're", "your": "your",
            "zealous": "zealous", "zucchini": "zucchini",
        }
    
    def correct(self, text: str) -> str:
        """
        Correct spelling of a word using fuzzy matching and common misspellings.
        """
        if not text:
            return text
        
        # Remove punctuation for matching but preserve it
        original_text = text
        text_lower = text.lower().strip()
        
        # Check common misspellings first
        if text_lower in self.common_misspellings:
            corrected = self.common_misspellings[text_lower]
            # Preserve original case
            if text and text[0].isupper():
                corrected = corrected.capitalize()
            return corrected
        
        # If it's already a known word, return as is
        if text_lower in self.common_words:
            return text
        
        # Try fuzzy matching against common words
        # Only correct if we find a good match (80%+ similarity)
        match = process.extractOne(text_lower, self.common_words, scorer=fuzz.WRatio)
        
        if match:
            best_match, score, _ = match
            if score >= 80:  # High confidence threshold
                # Preserve original case
                if text and text[0].isupper():
                    best_match = best_match.capitalize()
                return best_match
        
        # If no good match found, return original (might be a proper noun or unknown word)
        return text

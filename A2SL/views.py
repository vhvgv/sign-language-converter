from django.shortcuts import render
from django.http import HttpResponse
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import json
from django.contrib.staticfiles import finders
import urllib.request
import urllib.parse


# ── Auto-detect and translate to English ─────────────────────────────────────
def _translate_to_english(text):
    if not text:
        return text, "en", None
    try:
        params = urllib.parse.urlencode({
            "client": "gtx",
            "sl": "auto",
            "tl": "en",
            "dt": "t",
            "q": text,
        })
        url = "https://translate.googleapis.com/translate_a/single?" + params
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        translated    = "".join(part[0] for part in data[0] if part[0])
        detected_lang = data[2] if len(data) > 2 else "unknown"
        return translated.strip(), detected_lang, None
    except Exception as exc:
        return text, "unknown", f"⚠ Auto-translation unavailable: {str(exc)}"


# ── NLP pipeline ──────────────────────────────────────────────────────────────
STOP_WORDS = set([
    "mightn't", 're', 'wasn', 'wouldn', 'be', 'has', 'that', 'does',
    'shouldn', 'do', "you've", 'off', 'for', "didn't", 'm', 'ain', 'haven',
    "weren't", 'are', "she's", "wasn't", 'its', "haven't", "wouldn't", 'don',
    'weren', 's', "you'd", "don't", 'doesn', "hadn't", 'is', 'was', "that'll",
    "should've", 'a', 'then', 'the', 'mustn', 'i', 'nor', 'as', "it's",
    "needn't", 'd', 'am', 'have', 'hasn', 'o', "aren't", "you'll",
    "couldn't", "you're", "mustn't", 'didn', "doesn't", 'll', 'an', 'hadn',
    'whom', 'y', "hasn't", 'itself', 'couldn', 'needn', "shan't", 'isn',
    'been', 'such', 'shan', "shouldn't", 'aren', 'being', 'were', 'did',
    'ma', 't', 'having', 'mightn', 've', "isn't", "won't",
])


def _nlp_pipeline(english_text):
    words  = word_tokenize(english_text.lower())
    words  = [w for w in words if any(c.isalnum() for c in w)]
    tagged = nltk.pos_tag(words)
    lr     = WordNetLemmatizer()

    lemmatized = []
    for w, (_, pos) in zip(words, tagged):
        if w in STOP_WORDS:
            continue
        if pos in ('VBG', 'VBD', 'VBZ', 'VBN', 'NN'):
            lemmatized.append(lr.lemmatize(w, pos='v'))
        elif pos in ('JJ', 'JJR', 'JJS', 'RBR', 'RBS'):
            lemmatized.append(lr.lemmatize(w, pos='a'))
        else:
            lemmatized.append(lr.lemmatize(w))

    final_tokens = []
    for w in lemmatized:
        if finders.find(w + ".mp4"):
            final_tokens.append(w)
        else:
            final_tokens.extend(list(w))
    return final_tokens


# ── Learning mode data ────────────────────────────────────────────────────────
# All words here EXACTLY match your MP4 filenames (case-sensitive as stored).
# "words" list = the MP4 filenames (without .mp4) played in sequence.

TWO_WORD_SETS = [
    # Greetings & Farewells
    {"display": "Hello Bye",        "words": ["Hello",   "Bye"]},
    {"display": "Hello You",        "words": ["Hello",   "You"]},
    {"display": "Welcome Here",     "words": ["Welcome", "Here"]},
    {"display": "Hello Again",      "words": ["Hello",   "Again"]},
    {"display": "Bye Now",          "words": ["Bye",     "Now"]},
    {"display": "Thank You",        "words": ["Thank",   "You"]},

    # Feelings
    {"display": "Feel Happy",       "words": ["Happy",   "You"]},
    {"display": "Feel Sad",         "words": ["Sad",     "You"]},
    {"display": "Feel Good",        "words": ["Good",    "You"]},
    {"display": "Feel Great",       "words": ["Great",   "You"]},
    {"display": "Feel Alone",       "words": ["Alone",   "You"]},
    {"display": "Feel Beautiful",   "words": ["Beautiful","You"]},

    # Actions
    {"display": "Go Home",          "words": ["Go",      "Home"]},
    {"display": "Come Here",        "words": ["Come",    "Here"]},
    {"display": "Stay Safe",        "words": ["Stay",    "Safe"]},
    {"display": "Walk More",        "words": ["Walk",    "More"]},
    {"display": "Study More",       "words": ["Study",   "More"]},
    {"display": "Learn Sign",       "words": ["Learn",   "Sign"]},
    {"display": "Sing More",        "words": ["Sing",    "More"]},
    {"display": "Eat More",         "words": ["Eat",     "More"]},
    {"display": "Help Me",          "words": ["Help",    "ME"]},
    {"display": "See You",          "words": ["See",     "You"]},
    {"display": "Keep Going",       "words": ["Keep",    "Go"]},

    # People & Pronouns
    {"display": "My Name",          "words": ["My",      "Name"]},
    {"display": "Your Name",        "words": ["Your",    "Name"]},
    {"display": "My Hand",          "words": ["My",      "Hand"]},
    {"display": "My Home",          "words": ["My",      "Home"]},
    {"display": "Your Work",        "words": ["Your",    "Work"]},
    {"display": "My Language",      "words": ["My",      "Language"]},

    # Questions
    {"display": "Who You",          "words": ["Who",     "You"]},
    {"display": "What Time",        "words": ["What",    "Time"]},
    {"display": "Where Home",       "words": ["Where",   "Home"]},
    {"display": "How You",          "words": ["How",     "You"]},
    {"display": "When Next",        "words": ["When",    "Next"]},
    {"display": "Why Not",          "words": ["Why",     "Not"]},

    # Adjectives
    {"display": "Good Work",        "words": ["Good",    "Work"]},
    {"display": "Best Day",         "words": ["Best",    "Day"]},
    {"display": "Better Way",       "words": ["Better",  "Way"]},
    {"display": "Great Work",       "words": ["Great",   "Work"]},
    {"display": "Pretty Good",      "words": ["Pretty",  "Good"]},
    {"display": "Beautiful World",  "words": ["Beautiful","World"]},

    # Places & Things
    {"display": "Go College",       "words": ["Go",      "College"]},
    {"display": "My College",       "words": ["My",      "College"]},
    {"display": "Computer Language","words": ["Computer","Language"]},
    {"display": "Sign Language",    "words": ["Sign",    "Language"]},
    {"display": "World Language",   "words": ["World",   "Language"]},
    {"display": "My Computer",      "words": ["My",      "Computer"]},
]

THREE_WORD_SETS = [
    # Greetings
    {"display": "Hello How You",        "words": ["Hello",    "How",      "You"]},
    {"display": "Welcome Here Again",   "words": ["Welcome",  "Here",     "Again"]},
    {"display": "Hello Bye Now",        "words": ["Hello",    "Bye",      "Now"]},
    {"display": "Thank You Again",      "words": ["Thank",    "You",      "Again"]},
    {"display": "See You Again",        "words": ["See",      "You",      "Again"]},
    {"display": "See You Next",         "words": ["See",      "You",      "Next"]},

    # About self
    {"display": "My Name Here",         "words": ["My",       "Name",     "Here"]},
    {"display": "I Am Happy",           "words": ["I",        "Happy",    "You"]},
    {"display": "I Am Here",            "words": ["I",        "Here",     "Now"]},
    {"display": "I Need Help",          "words": ["I",        "Help",     "ME"]},
    {"display": "I Go Home",            "words": ["I",        "Go",       "Home"]},
    {"display": "I Go College",         "words": ["I",        "Go",       "College"]},

    # Questions
    {"display": "What Your Name",       "words": ["What",     "Your",     "Name"]},
    {"display": "Where You Go",         "words": ["Where",    "You",      "Go"]},
    {"display": "How You Work",         "words": ["How",      "You",      "Work"]},
    {"display": "Why You Sad",          "words": ["Why",      "You",      "Sad"]},
    {"display": "When You Come",        "words": ["When",     "You",      "Come"]},
    {"display": "Who You Are",          "words": ["Who",      "You",      "Here"]},
    {"display": "What Time Now",        "words": ["What",     "Time",     "Now"]},
    {"display": "Where You Stay",       "words": ["Where",    "You",      "Stay"]},

    # Actions
    {"display": "Come Here Now",        "words": ["Come",     "Here",     "Now"]},
    {"display": "Go Home Now",          "words": ["Go",       "Home",     "Now"]},
    {"display": "Stay Here Safe",       "words": ["Stay",     "Here",     "Safe"]},
    {"display": "Walk More Day",        "words": ["Walk",     "More",     "Day"]},
    {"display": "Learn Sign Language",  "words": ["Learn",    "Sign",     "Language"]},
    {"display": "Study More Better",    "words": ["Study",    "More",     "Better"]},
    {"display": "Help ME Now",          "words": ["Help",     "ME",       "Now"]},
    {"display": "Keep Going More",      "words": ["Keep",     "Go",       "More"]},

    # Feelings
    {"display": "I Feel Happy",         "words": ["I",        "Happy",    "Good"]},
    {"display": "You Look Beautiful",   "words": ["You",      "Beautiful","Good"]},
    {"display": "Feel Good Today",      "words": ["Good",     "Day",      "You"]},
    {"display": "Not Sad Alone",        "words": ["Not",      "Sad",      "Alone"]},
    {"display": "Best Day Again",       "words": ["Best",     "Day",      "Again"]},
    {"display": "Great Work Again",     "words": ["Great",    "Work",     "Again"]},

    # Places & Things
    {"display": "Sign Language World",  "words": ["Sign",     "Language", "World"]},
    {"display": "My Computer Work",     "words": ["My",       "Computer", "Work"]},
    {"display": "Go College Now",       "words": ["Go",       "College",  "Now"]},
    {"display": "My Home Here",         "words": ["My",       "Home",     "Here"]},
    {"display": "World Best Language",  "words": ["World",    "Best",     "Language"]},
    {"display": "Good Work Day",        "words": ["Good",     "Work",     "Day"]},
]


# ── Views ─────────────────────────────────────────────────────────────────────
def home_view(request):
    return render(request, 'home.html')


def learn_view(request):
    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    return render(request, 'learn.html', {
        'alphabet':             alphabet,
        'two_word_sets':        TWO_WORD_SETS,
        'three_word_sets':      THREE_WORD_SETS,
        'two_word_sets_json':   json.dumps(TWO_WORD_SETS),
        'three_word_sets_json': json.dumps(THREE_WORD_SETS),
    })


def animation_view(request):
    if request.method == 'POST':
        raw_text = request.POST.get('sen', '').strip()
        english_text, detected_lang, trans_error = _translate_to_english(raw_text)
        words = _nlp_pipeline(english_text)
        return render(request, 'animation.html', {
            'words':         words,
            'text':          raw_text,
            'english_text':  english_text,
            'detected_lang': detected_lang,
            'trans_error':   trans_error,
        })
    return render(request, 'animation.html')


def manifest_view(request):
    body = json.dumps({
        "id": "/", "name": "சைகை", "short_name": "சைகை",
        "description": "Multilingual speech to sign animation workspace",
        "start_url": "/", "scope": "/", "display": "standalone",
        "display_override": ["standalone", "minimal-ui"],
        "background_color": "#070b1a", "theme_color": "#101a36",
        "lang": "en-US",
        "icons": [
            {"src": "/static/logo.jpg", "sizes": s, "type": "image/jpeg", "purpose": "any"}
            for s in ["192x192", "512x512", "180x180"]
        ],
    }, ensure_ascii=False, indent=2)
    r = HttpResponse(body, content_type='application/manifest+json')
    r['Cache-Control'] = 'no-cache'
    return r


def service_worker_view(request):
    body = """const CACHE_NAME = 'gesturestream-v3';
const ASSETS = ['/static/logo.jpg', '/static/mic3.png'];
self.addEventListener('install',   e => e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(ASSETS))));
self.addEventListener('activate',  e => e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k=>k!==CACHE_NAME).map(k=>caches.delete(k))))));
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  if (e.request.mode === 'navigate' || (e.request.headers.get('accept')||'').includes('text/html')) return;
  e.respondWith(caches.match(e.request).then(c => c || fetch(e.request)));
});"""
    r = HttpResponse(body, content_type='application/javascript')
    r['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return r
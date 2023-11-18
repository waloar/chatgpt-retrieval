import nltk
from nltk.chat.util import Chat, reflections
pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, How are you today?"]
    ],
    [
        r"hi|hey|hello",
        ["Hello", "Hey there"]
    ],
    [
        r"what is your name ?",
        ["I'm a chatbot and I don't have a name"]
    ],
    [
        r"how are you ?",
        ["I'm fine"]
    ],
    [
        r"sorry (.*)",
        ["Its alright", "Its OK, never mind"]
    ],
    [
        r"i am fine",
        ["Great to hear that", "Awesome!"]
    ],
    [
        r"quit",
        ["Bye bye, take care. See you soon :) "]
    ],
]
chatbot = Chat(pairs, reflections)
chatbot.converse()

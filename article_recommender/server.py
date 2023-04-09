# Launch with
#
# gunicorn --threads 4 -b 0.0.0.0:5000 --access-logfile server.log --timeout 60 server:app glove.6B.300d.txt bbc

from flask import Flask, render_template
from doc2vec import *
import sys

app = Flask(__name__)

@app.route("/")
def articles():
    """Show a list of article titles"""
    return render_template('articles.html', articles=artlist)


@app.route("/article/<topic>/<filename>")
def article(topic,filename):
    """
    Show an article with relative path filename. Assumes the BBC structure of
    topic/filename.txt so our URLs follow that.
    """
    filename = topic + '/' + filename
    docs = [a for a in artlist if a[0]==filename]
    doc = docs[0]
    title = doc[1]
    text = doc[2].strip()
    paragraphs = text.split('\n\n')
    seealso = recommended(doc, artlist, 5)
    return render_template('article.html', title=title, paragraphs=paragraphs, recommended=seealso)


# initialization
i=0
# i = sys.argv.index('server:app')
glove_filename = sys.argv[i+1]
articles_dirname = sys.argv[i+2]

gloves = load_glove(glove_filename)
artlist = load_articles(articles_dirname, gloves)

app.run() # use gunicorn not this dev server
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 21:46:58 2021

@author: linxiangling
"""

from pymongo import MongoClient
import pytz

# DB = MongoClient('mongodb+srv://pqamanager:pass123word@pqacluster0.umi6y.mongodb.net/test')['PQAbot']
DB = MongoClient('mongodb+srv://psabot:psabot2022@psabot.dt5vbbu.mongodb.net/test')['PSAbot']

TAG_COLLECTION = DB['Tag']
USER_COLLECTION = DB['User']
INNER_POST_COLLECTION = DB['InnerPost']
OUTER_DATA_COLLECTION = DB['OuterData']
OUTER_DATA_CACHE_COLLECTION = DB['OuterDataCache']
FAQ_DATA_COLLECTION = DB['FAQData']
CHAT_DATA_COLLECTION = DB['ChatData']
TOPIC_DATA_COLLECTION = DB['Topic']
TEST_DATA = DB['TestData']
TEST_OUTER_DATA = DB['TestOuterData']
# specify time zone
pacific = pytz.timezone('Asia/Taipei')




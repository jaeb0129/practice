import { useState, useEffect, useRef } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// WBC 2026 해설 도우미 (뉴스 버전) — CSV 공식 명단 + MLB StatsAPI + 관련 기사
// 선수 클릭 → WBC/MLB 성적 + Anthropic web_search로 관련 기사 헤드라인+링크
// ─────────────────────────────────────────────────────────────────────────────

const WBC_POOLS = {
  "Pool A": { location:"San Juan",  teams:["Canada","Colombia","Cuba","Panama","Puerto Rico"] },
  "Pool B": { location:"Houston",   teams:["Brazil","Great Britain","Italy","Mexico","USA"] },
  "Pool C": { location:"Tokyo",     teams:["Australia","Chinese Taipei","Czechia","Japan","Korea"] },
  "Pool D": { location:"Miami",     teams:["Dominican Rep.","Israel","Netherlands","Nicaragua","Venezuela"] },
};

const TEAM_FLAG = {
  "USA":"🇺🇸","Japan":"🇯🇵","Korea":"🇰🇷","Mexico":"🇲🇽","Dominican Rep.":"🇩🇴",
  "Venezuela":"🇻🇪","Puerto Rico":"🇵🇷","Cuba":"🇨🇺","Canada":"🇨🇦","Italy":"🇮🇹",
  "Netherlands":"🇳🇱","Colombia":"🇨🇴","Panama":"🇵🇦","Israel":"🇮🇱","Australia":"🇦🇺",
  "Chinese Taipei":"🇹🇼","Great Britain":"🇬🇧","Brazil":"🇧🇷","Nicaragua":"🇳🇮","Czechia":"🇨🇿",
};

const ALL_PLAYERS = [
  {id:1,name:"Aaron Whitefield",team:"Australia",pool:"Pool C",pos:"OF",mlbId:664334,bats:"R",throws:"R",num:2,dob:"1996-09-02",age:30,h:193,w:95,isPitcher:false,wbcStats:{PA:16,AB:15,H:3,HR:0,R:1,RBI:0,BB:1,SO:3,SB:2,AVG:0.2,OBP:0.25}},
  {id:2,name:"Alex Hall",team:"Australia",pool:"Pool C",pos:"C",mlbId:673064,bats:"S",throws:"R",num:10,dob:"1999-06-08",age:27,h:178,w:93,isPitcher:false,wbcStats:{PA:16,AB:15,H:4,HR:2,R:2,RBI:2,BB:1,SO:3,SB:0,AVG:0.267,OBP:0.312}},
  {id:3,name:"Alex Wells",team:"Australia",pool:"Pool C",pos:"P",mlbId:649144,bats:"L",throws:"L",num:8,dob:"1997-02-27",age:29,h:185,w:88,isPitcher:true,wbcStats:{G:2,GS:1,W:0,L:0,H:0,SV:0,IP:6.0,SO:9,BB:2,HA:3,HR:0,ER:2,ERA:3.0,WHIP:0.83,Kpct:39.1,BBpct:8.7}},
  {id:4,name:"Blake Townsend",team:"Australia",pool:"Pool C",pos:"P",mlbId:678620,bats:"L",throws:"L",num:54,dob:"2001-04-05",age:25,h:193,w:111,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:2.1,SO:2,BB:1,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.43,Kpct:28.6,BBpct:14.3}},
  {id:5,name:"Chris Burke",team:"Australia",pool:"Pool C",pos:"OF",mlbId:678521,bats:"L",throws:"R",num:34,dob:"2001-08-16",age:25,h:178,w:82,isPitcher:false,wbcStats:{PA:10,AB:8,H:2,HR:0,R:0,RBI:0,BB:2,SO:2,SB:0,AVG:0.25,OBP:0.4}},
  {id:6,name:"Coen Wynne",team:"Australia",pool:"Pool C",pos:"P",mlbId:807650,bats:"R",throws:"R",num:38,dob:"1999-01-25",age:27,h:193,w:86,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:2,SV:0,IP:3.0,SO:0,BB:1,HA:3,HR:0,ER:2,ERA:6.0,WHIP:1.33,Kpct:0.0,BBpct:7.7}},
  {id:7,name:"Connor MacDonald",team:"Australia",pool:"Pool C",pos:"P",mlbId:628015,bats:"R",throws:"R",num:39,dob:"1996-02-27",age:30,h:196,w:91,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:3.0,SO:1,BB:3,HA:1,HR:0,ER:0,ERA:0.0,WHIP:1.33,Kpct:7.7,BBpct:23.1}},
  {id:8,name:"Cooper Morgan",team:"Australia",pool:"Pool C",pos:"P",mlbId:692707,bats:"L",throws:"L",num:65,dob:"2001-11-08",age:25,h:0,w:0,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:0.2,SO:0,BB:0,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.0,Kpct:0.0,BBpct:0.0}},
  {id:9,name:"Curtis Mead",team:"Australia",pool:"Pool C",pos:"1B",mlbId:678554,bats:"R",throws:"R",num:16,dob:"2000-10-26",age:26,h:185,w:102,isPitcher:false,wbcStats:{PA:16,AB:14,H:5,HR:1,R:1,RBI:3,BB:1,SO:3,SB:0,AVG:0.357,OBP:0.4}},
  {id:10,name:"George Callil",team:"Australia",pool:"Pool C",pos:"SS",mlbId:666838,bats:"R",throws:"R",num:28,dob:"1997-07-22",age:29,h:193,w:91,isPitcher:false,wbcStats:{PA:0,AB:0,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0}},
  {id:11,name:"Jack O'Loughlin",team:"Australia",pool:"Pool C",pos:"P",mlbId:672552,bats:"L",throws:"L",num:37,dob:"2000-03-14",age:26,h:196,w:101,isPitcher:true,wbcStats:{G:2,GS:0,W:1,L:0,H:0,SV:0,IP:6.1,SO:5,BB:1,HA:4,HR:0,ER:0,ERA:0.0,WHIP:0.79,Kpct:20.0,BBpct:4.0}},
  {id:12,name:"Jarryd Dale",team:"Australia",pool:"Pool C",pos:"SS",mlbId:673062,bats:"R",throws:"R",num:43,dob:"2000-09-11",age:26,h:188,w:91,isPitcher:false,wbcStats:{PA:16,AB:15,H:4,HR:0,R:1,RBI:0,BB:1,SO:3,SB:0,AVG:0.267,OBP:0.312}},
  {id:13,name:"Jon Kennedy",team:"Australia",pool:"Pool C",pos:"P",mlbId:626943,bats:"L",throws:"L",num:55,dob:"1994-09-20",age:32,h:203,w:111,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:1,H:0,SV:1,IP:4.1,SO:1,BB:2,HA:3,HR:1,ER:2,ERA:4.15,WHIP:1.15,Kpct:5.3,BBpct:10.5}},
  {id:14,name:"Josh Hendrickson",team:"Australia",pool:"Pool C",pos:"P",mlbId:681973,bats:"L",throws:"L",num:44,dob:"1997-09-18",age:29,h:193,w:98,isPitcher:true,wbcStats:{G:1,GS:1,W:1,L:0,H:0,SV:0,IP:3.0,SO:2,BB:1,HA:2,HR:0,ER:1,ERA:3.0,WHIP:1.0,Kpct:16.7,BBpct:8.3}},
  {id:15,name:"Kieren Hall",team:"Australia",pool:"Pool C",pos:"P",mlbId:816332,bats:"R",throws:"R",num:46,dob:"2001-05-10",age:25,h:183,w:88,isPitcher:true,wbcStats:{G:0,GS:0,W:0,L:0,H:0,SV:0,IP:0.0,SO:0,BB:0,HA:0,HR:0,ER:0}},
  {id:16,name:"Ky Hampton",team:"Australia",pool:"Pool C",pos:"P",mlbId:673132,bats:"R",throws:"R",num:26,dob:"2000-10-04",age:26,h:185,w:70,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:1.1,SO:2,BB:4,HA:2,HR:0,ER:2,ERA:13.5,WHIP:4.5,Kpct:20.0,BBpct:40.0}},
  {id:17,name:"Lachlan Wells",team:"Australia",pool:"Pool C",pos:"P",mlbId:649143,bats:"L",throws:"L",num:19,dob:"1997-02-27",age:29,h:185,w:84,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:1,H:0,SV:0,IP:1.2,SO:0,BB:2,HA:2,HR:1,ER:2,ERA:10.8,WHIP:2.4,Kpct:0.0,BBpct:22.2}},
  {id:18,name:"Logan Wade",team:"Australia",pool:"Pool C",pos:"2B",mlbId:623921,bats:"S",throws:"R",num:4,dob:"1991-11-13",age:35,h:185,w:86,isPitcher:false,wbcStats:{PA:1,AB:1,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0,AVG:0.0,OBP:0.0}},
  {id:19,name:"Max Durrington",team:"Australia",pool:"Pool C",pos:"OF",mlbId:828589,bats:"L",throws:"R",num:36,dob:"2007-02-13",age:19,h:175,w:75,isPitcher:false,wbcStats:{PA:0,AB:0,H:0,HR:0,R:1,RBI:0,BB:0,SO:0,SB:0}},
  {id:20,name:"Mitch Neunborn",team:"Australia",pool:"Pool C",pos:"P",mlbId:666748,bats:"R",throws:"R",num:22,dob:"1997-06-27",age:29,h:183,w:86,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.2,SO:2,BB:1,HA:1,HR:0,ER:0,ERA:0.0,WHIP:1.2,Kpct:28.6,BBpct:14.3}},
  {id:21,name:"Mitchell Edwards",team:"Australia",pool:"Pool C",pos:"C",mlbId:673134,bats:"S",throws:"R",num:1,dob:"1999-08-01",age:27,h:180,w:91,isPitcher:false,wbcStats:{PA:0,AB:0,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0}},
  {id:22,name:"Rixon Wingrove",team:"Australia",pool:"Pool C",pos:"1B",mlbId:678534,bats:"L",throws:"R",num:52,dob:"2000-05-23",age:26,h:193,w:118,isPitcher:false,wbcStats:{PA:15,AB:14,H:2,HR:1,R:2,RBI:1,BB:0,SO:4,SB:0,AVG:0.143,OBP:0.143}},
  {id:23,name:"Robbie Glendinning",team:"Australia",pool:"Pool C",pos:"3B",mlbId:674671,bats:"R",throws:"R",num:6,dob:"1995-10-06",age:31,h:183,w:89,isPitcher:false,wbcStats:{PA:6,AB:5,H:2,HR:1,R:1,RBI:1,BB:1,SO:2,SB:0,AVG:0.4,OBP:0.5}},
  {id:24,name:"Robbie Perkins",team:"Australia",pool:"Pool C",pos:"C",mlbId:564434,bats:"R",throws:"R",num:9,dob:"1994-05-29",age:32,h:191,w:90,isPitcher:false,wbcStats:{PA:15,AB:13,H:2,HR:1,R:1,RBI:3,BB:2,SO:6,SB:0,AVG:0.154,OBP:0.267}},
  {id:25,name:"Sam Holland",team:"Australia",pool:"Pool C",pos:"P",mlbId:627697,bats:"R",throws:"R",num:40,dob:"1994-02-20",age:32,h:193,w:91,isPitcher:true,wbcStats:{G:0,GS:0,W:0,L:0,H:0,SV:0,IP:0.0,SO:0,BB:0,HA:0,HR:0,ER:0}},
  {id:26,name:"Tim Kennelly",team:"Australia",pool:"Pool C",pos:"OF",mlbId:470491,bats:"R",throws:"R",num:23,dob:"1986-12-05",age:40,h:183,w:82,isPitcher:false,wbcStats:{PA:13,AB:10,H:2,HR:0,R:1,RBI:0,BB:2,SO:1,SB:0,AVG:0.2,OBP:0.333}},
  {id:27,name:"Todd Van Steensel",team:"Australia",pool:"Pool C",pos:"P",mlbId:573667,bats:"R",throws:"R",num:21,dob:"1991-01-14",age:35,h:185,w:104,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:1,SV:0,IP:1.0,SO:0,BB:1,HA:0,HR:0,ER:0,ERA:0.0,WHIP:1.0,Kpct:0.0,BBpct:33.3}},
  {id:28,name:"Travis Bazzana",team:"Australia",pool:"Pool C",pos:"2B",mlbId:683953,bats:"L",throws:"R",num:64,dob:"2002-08-28",age:24,h:180,w:90,isPitcher:false,wbcStats:{PA:17,AB:16,H:3,HR:1,R:2,RBI:2,BB:1,SO:3,SB:0,AVG:0.188,OBP:0.235}},
  {id:29,name:"Ulrich Bojarski",team:"Australia",pool:"Pool C",pos:"RF",mlbId:672531,bats:"R",throws:"R",num:25,dob:"1998-09-15",age:28,h:191,w:100,isPitcher:false,wbcStats:{PA:0,AB:0,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0}},
  {id:30,name:"Warwick Saupold",team:"Australia",pool:"Pool C",pos:"P",mlbId:599683,bats:"R",throws:"R",num:30,dob:"1990-01-16",age:36,h:188,w:101,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:0.2,SO:0,BB:2,HA:1,HR:0,ER:0,ERA:0.0,WHIP:4.5,Kpct:0.0,BBpct:40.0}},
  {id:31,name:"An Ko Lin",team:"Chinese Taipei",pool:"Pool C",pos:"OF",mlbId:838360,bats:"L",throws:"L",num:77,dob:"1997-05-19",age:29,h:183,w:90,isPitcher:false,wbcStats:{PA:15,AB:15,H:1,HR:0,R:0,RBI:0,BB:0,SO:4,SB:0,AVG:0.067,OBP:0.067}},
  {id:32,name:"Chen Zhong Ao Zhuang",team:"Chinese Taipei",pool:"Pool C",pos:"P",mlbId:800018,bats:"R",throws:"R",num:48,dob:"2000-08-25",age:26,h:185,w:90,isPitcher:true,wbcStats:{G:1,GS:1,W:1,L:0,H:0,SV:0,IP:2.2,SO:4,BB:1,HA:2,HR:0,ER:0,ERA:0.0,WHIP:1.12,Kpct:33.3,BBpct:8.3}},
  {id:33,name:"Chen Wei Chen",team:"Chinese Taipei",pool:"Pool C",pos:"OF",mlbId:808992,bats:"L",throws:"R",num:98,dob:"1997-12-12",age:29,h:180,w:72,isPitcher:false,wbcStats:{PA:15,AB:13,H:3,HR:0,R:1,RBI:3,BB:1,SO:3,SB:2,AVG:0.231,OBP:0.286}},
  {id:34,name:"Cheng Hui Sung",team:"Chinese Taipei",pool:"Pool C",pos:"OF",mlbId:830723,bats:"R",throws:"R",num:88,dob:"2002-08-14",age:24,h:183,w:78,isPitcher:false,wbcStats:{PA:7,AB:6,H:0,HR:0,R:1,RBI:0,BB:1,SO:4,SB:0,AVG:0.0,OBP:0.143}},
  {id:35,name:"Cheng Yu Chang",team:"Chinese Taipei",pool:"Pool C",pos:"IF",mlbId:839138,bats:"L",throws:"R",num:9,dob:"2000-06-08",age:26,h:178,w:70,isPitcher:false,wbcStats:{PA:2,AB:2,H:0,HR:0,R:1,RBI:0,BB:0,SO:0,SB:0,AVG:0.0,OBP:0.0}},
  {id:36,name:"Chieh Hsien Chen",team:"Chinese Taipei",pool:"Pool C",pos:"OF",mlbId:808993,bats:"L",throws:"R",num:24,dob:"1994-01-07",age:32,h:173,w:78,isPitcher:false,wbcStats:{PA:3,AB:1,H:0,HR:0,R:1,RBI:0,BB:1,SO:0,SB:0,AVG:0.0,OBP:0.5}},
  {id:37,name:"Chih Wei Hu",team:"Chinese Taipei",pool:"Pool C",pos:"P",mlbId:629496,bats:"R",throws:"R",num:58,dob:"1993-11-04",age:33,h:183,w:110,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:0.1,SO:0,BB:1,HA:3,HR:0,ER:2,ERA:54.0,WHIP:12.0,Kpct:0.0,BBpct:20.0}},
  {id:38,name:"Hao Chun Cheng",team:"Chinese Taipei",pool:"Pool C",pos:"P",mlbId:692059,bats:"R",throws:"R",num:47,dob:"1997-09-17",age:29,h:191,w:93,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:1,H:0,SV:0,IP:1.2,SO:2,BB:4,HA:5,HR:1,ER:8,ERA:43.2,WHIP:5.4,Kpct:13.3,BBpct:26.7}},
  {id:39,name:"Jo Hsi Hsu",team:"Chinese Taipei",pool:"Pool C",pos:"P",mlbId:830717,bats:"R",throws:"R",num:0,dob:"2000-11-01",age:26,h:175,w:76,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:4.0,SO:3,BB:0,HA:2,HR:0,ER:0,ERA:0.0,WHIP:0.5,Kpct:21.4,BBpct:0.0}},
  {id:40,name:"Jonathon Long",team:"Chinese Taipei",pool:"Pool C",pos:"1B",mlbId:675085,bats:"R",throws:"R",num:22,dob:"2002-01-20",age:24,h:183,w:95,isPitcher:false,wbcStats:{PA:0,AB:0,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0}},
  {id:41,name:"Jun Wei Zhang",team:"Chinese Taipei",pool:"Pool C",pos:"P",mlbId:838362,bats:"R",throws:"R",num:37,dob:"2005-11-14",age:21,h:178,w:82,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:2.2,SO:0,BB:2,HA:1,HR:0,ER:0,ERA:0.0,WHIP:1.12,Kpct:0.0,BBpct:20.0}},
  {id:42,name:"Jyun Yue Tseng",team:"Chinese Taipei",pool:"Pool C",pos:"P",mlbId:809004,bats:"R",throws:"R",num:60,dob:"2001-11-07",age:25,h:173,w:68,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:1,IP:2.0,SO:2,BB:0,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.0,Kpct:33.3,BBpct:0.0}},
  {id:43,name:"Kai Wei Lin",team:"Chinese Taipei",pool:"Pool C",pos:"P",mlbId:666020,bats:"R",throws:"R",num:0,dob:"1996-03-19",age:30,h:178,w:79,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.2,SO:1,BB:0,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.0,Kpct:25.0,BBpct:0.0}},
  {id:44,name:"Kuan Yu Chen",team:"Chinese Taipei",pool:"Pool C",pos:"P",mlbId:611563,bats:"L",throws:"L",num:20,dob:"1990-10-29",age:36,h:175,w:85,isPitcher:true,wbcStats:{G:0,GS:0,W:0,L:0,H:0,SV:0,IP:0.0,SO:0,BB:0,HA:0,HR:0,ER:0}},
  {id:45,name:"Kun Yu Chiang",team:"Chinese Taipei",pool:"Pool C",pos:"SS",mlbId:808997,bats:"R",throws:"R",num:90,dob:"2000-07-04",age:26,h:173,w:72,isPitcher:false,wbcStats:{PA:14,AB:12,H:2,HR:0,R:2,RBI:2,BB:1,SO:2,SB:1,AVG:0.167,OBP:0.231}},
  {id:46,name:"Kungkuan Giljegiljaw",team:"Chinese Taipei",pool:"Pool C",pos:"C",mlbId:644375,bats:"R",throws:"R",num:4,dob:"1994-03-13",age:32,h:180,w:100,isPitcher:false,wbcStats:{PA:12,AB:9,H:1,HR:0,R:0,RBI:0,BB:2,SO:4,SB:0,AVG:0.111,OBP:0.273}},
  {id:47,name:"Lyle Lin",team:"Chinese Taipei",pool:"Pool C",pos:"C",mlbId:670076,bats:"R",throws:"R",num:27,dob:"1997-06-26",age:29,h:183,w:91,isPitcher:false,wbcStats:{PA:10,AB:7,H:1,HR:0,R:2,RBI:0,BB:1,SO:1,SB:0,AVG:0.143,OBP:0.25}},
  {id:48,name:"Nien Ting Wu",team:"Chinese Taipei",pool:"Pool C",pos:"1B",mlbId:809006,bats:"L",throws:"R",num:39,dob:"1993-06-07",age:33,h:178,w:75,isPitcher:false,wbcStats:{PA:11,AB:11,H:2,HR:0,R:0,RBI:0,BB:0,SO:3,SB:0,AVG:0.182,OBP:0.182}},
  {id:49,name:"Po Yu Chen",team:"Chinese Taipei",pool:"Pool C",pos:"P",mlbId:696040,bats:"L",throws:"R",num:44,dob:"2001-10-02",age:25,h:185,w:91,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:1,H:0,SV:0,IP:1.0,SO:2,BB:1,HA:1,HR:1,ER:2,ERA:18.0,WHIP:2.0,Kpct:33.3,BBpct:16.7}},
  {id:50,name:"Ruei Yang Gu Lin",team:"Chinese Taipei",pool:"Pool C",pos:"P",mlbId:830716,bats:"R",throws:"R",num:11,dob:"2000-06-12",age:26,h:185,w:81,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:4.0,SO:1,BB:1,HA:2,HR:0,ER:1,ERA:2.25,WHIP:0.75,Kpct:7.1,BBpct:7.1}},
  {id:51,name:"Shao Hung Chiang",team:"Chinese Taipei",pool:"Pool C",pos:"C",mlbId:830725,bats:"R",throws:"R",num:63,dob:"1997-07-13",age:29,h:180,w:100,isPitcher:false,wbcStats:{PA:4,AB:3,H:0,HR:0,R:0,RBI:0,BB:0,SO:1,SB:0,AVG:0.0,OBP:0.0}},
  {id:52,name:"Shih Hsiang Lin",team:"Chinese Taipei",pool:"Pool C",pos:"P",mlbId:838361,bats:"L",throws:"R",num:12,dob:"2001-07-31",age:25,h:180,w:83,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.1,SO:0,BB:0,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.75,Kpct:0.0,BBpct:0.0}},
  {id:53,name:"Stuart Fairchild",team:"Chinese Taipei",pool:"Pool C",pos:"CF",mlbId:656413,bats:"R",throws:"R",num:17,dob:"1996-03-17",age:30,h:180,w:93,isPitcher:false,wbcStats:{PA:16,AB:12,H:3,HR:2,R:5,RBI:6,BB:4,SO:5,SB:3,AVG:0.25,OBP:0.438}},
  {id:54,name:"Tsung Che Cheng",team:"Chinese Taipei",pool:"Pool C",pos:"SS",mlbId:691907,bats:"L",throws:"R",num:1,dob:"2001-07-26",age:25,h:173,w:82,isPitcher:false,wbcStats:{PA:15,AB:9,H:3,HR:1,R:4,RBI:2,BB:5,SO:1,SB:4,AVG:0.333,OBP:0.571}},
  {id:55,name:"Tzu Chen Sha",team:"Chinese Taipei",pool:"Pool C",pos:"P",mlbId:809223,bats:"R",throws:"R",num:92,dob:"2003-10-15",age:23,h:188,w:75,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:2.0,SO:1,BB:0,HA:4,HR:0,ER:3,ERA:13.5,WHIP:2.0,Kpct:10.0,BBpct:0.0}},
  {id:56,name:"Tzu Wei Lin",team:"Chinese Taipei",pool:"Pool C",pos:"SS",mlbId:624407,bats:"L",throws:"R",num:15,dob:"1994-02-15",age:32,h:175,w:82,isPitcher:false,wbcStats:{PA:3,AB:3,H:0,HR:0,R:0,RBI:0,BB:0,SO:1,SB:0,AVG:0.0,OBP:0.0}},
  {id:57,name:"Wei En Lin",team:"Chinese Taipei",pool:"Pool C",pos:"P",mlbId:827734,bats:"L",throws:"L",num:42,dob:"2005-11-04",age:21,h:188,w:81,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:2.1,SO:4,BB:2,HA:1,HR:1,ER:2,ERA:7.71,WHIP:1.29,Kpct:44.4,BBpct:22.2}},
  {id:58,name:"Yi Chang",team:"Chinese Taipei",pool:"Pool C",pos:"P",mlbId:838359,bats:"R",throws:"R",num:19,dob:"1994-02-26",age:32,h:183,w:84,isPitcher:true,wbcStats:{G:2,GS:0,W:1,L:0,H:0,SV:0,IP:1.2,SO:0,BB:1,HA:2,HR:1,ER:1,ERA:5.4,WHIP:1.8,Kpct:0.0,BBpct:12.5}},
  {id:59,name:"Yi Lei Sun",team:"Chinese Taipei",pool:"Pool C",pos:"P",mlbId:806844,bats:"L",throws:"R",num:96,dob:"2005-02-10",age:21,h:183,w:82,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:2.1,SO:3,BB:3,HA:2,HR:0,ER:1,ERA:3.86,WHIP:2.14,Kpct:23.1,BBpct:23.1}},
  {id:60,name:"Yu Chang",team:"Chinese Taipei",pool:"Pool C",pos:"SS",mlbId:644374,bats:"R",throws:"R",num:18,dob:"1995-08-18",age:31,h:185,w:95,isPitcher:false,wbcStats:{PA:16,AB:15,H:6,HR:1,R:2,RBI:5,BB:1,SO:2,SB:0,AVG:0.4,OBP:0.438}},
  {id:61,name:"Yu Min Lin",team:"Chinese Taipei",pool:"Pool C",pos:"P",mlbId:801179,bats:"L",throws:"L",num:45,dob:"2003-07-12",age:23,h:180,w:73,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:2.1,SO:3,BB:0,HA:2,HR:0,ER:0,ERA:0.0,WHIP:0.86,Kpct:33.3,BBpct:0.0}},
  {id:62,name:"Angel Obando",team:"Nicaragua",pool:"Pool D",pos:"P",mlbId:679752,bats:"R",throws:"R",num:2,dob:"1999-01-19",age:27,h:180,w:81,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:2,H:0,SV:0,IP:4.2,SO:4,BB:3,HA:6,HR:1,ER:6,ERA:11.57,WHIP:1.93,Kpct:17.4,BBpct:13.0}},
  {id:63,name:"Benjamin Alegria",team:"Nicaragua",pool:"Pool D",pos:"2B",mlbId:664342,bats:"R",throws:"R",num:1,dob:"1997-08-06",age:29,h:178,w:75,isPitcher:false,wbcStats:{PA:9,AB:9,H:4,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0,AVG:0.444,OBP:0.444}},
  {id:64,name:"Brandon Leyton",team:"Nicaragua",pool:"Pool D",pos:"SS",mlbId:667351,bats:"R",throws:"R",num:5,dob:"1998-12-17",age:28,h:178,w:75,isPitcher:false,wbcStats:{PA:5,AB:5,H:1,HR:1,R:1,RBI:1,BB:0,SO:1,SB:0,AVG:0.2,OBP:0.2}},
  {id:65,name:"Bryan Torres",team:"Nicaragua",pool:"Pool D",pos:"P",mlbId:678897,bats:"R",throws:"R",num:78,dob:"2001-04-12",age:25,h:188,w:82,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:2.2,SO:0,BB:1,HA:3,HR:0,ER:1,ERA:3.38,WHIP:1.5,Kpct:0.0,BBpct:7.7}},
  {id:66,name:"Carlos Rodriguez",team:"Nicaragua",pool:"Pool D",pos:"P",mlbId:692230,bats:"R",throws:"R",num:27,dob:"2001-11-27",age:25,h:180,w:92,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:1,H:0,SV:0,IP:4.0,SO:4,BB:2,HA:2,HR:0,ER:1,ERA:2.25,WHIP:1.0,Kpct:25.0,BBpct:12.5}},
  {id:67,name:"Carlos Teller",team:"Nicaragua",pool:"Pool D",pos:"P",mlbId:469191,bats:"L",throws:"L",num:37,dob:"1986-10-03",age:40,h:180,w:82,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:3.0,SO:3,BB:1,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.67,Kpct:27.3,BBpct:9.1}},
  {id:68,name:"Chase Dawson",team:"Nicaragua",pool:"Pool D",pos:"OF",mlbId:689209,bats:"L",throws:"R",num:15,dob:"1997-06-12",age:29,h:175,w:84,isPitcher:false,wbcStats:{PA:23,AB:22,H:2,HR:0,R:3,RBI:0,BB:0,SO:4,SB:0,AVG:0.091,OBP:0.091}},
  {id:69,name:"Cheslor Cuthbert",team:"Nicaragua",pool:"Pool D",pos:"3B",mlbId:596144,bats:"R",throws:"R",num:24,dob:"1992-11-16",age:34,h:185,w:86,isPitcher:false,wbcStats:{PA:21,AB:19,H:3,HR:0,R:0,RBI:1,BB:2,SO:8,SB:0,AVG:0.158,OBP:0.238}},
  {id:70,name:"Cristhian Sandoval",team:"Nicaragua",pool:"Pool D",pos:"OF",mlbId:830698,bats:"R",throws:"R",num:95,dob:"2000-05-09",age:26,h:180,w:82,isPitcher:false,wbcStats:{PA:14,AB:12,H:3,HR:0,R:1,RBI:0,BB:1,SO:4,SB:0,AVG:0.25,OBP:0.308}},
  {id:71,name:"Danilo Bermudez",team:"Nicaragua",pool:"Pool D",pos:"P",mlbId:807358,bats:"L",throws:"L",num:42,dob:"1999-03-22",age:27,h:178,w:85,isPitcher:true,wbcStats:{G:3,GS:1,W:0,L:1,H:0,SV:0,IP:7.0,SO:3,BB:4,HA:3,HR:1,ER:3,ERA:3.86,WHIP:1.0,Kpct:11.1,BBpct:14.8}},
  {id:72,name:"Dilmer Mejia",team:"Nicaragua",pool:"Pool D",pos:"P",mlbId:650356,bats:"L",throws:"L",num:32,dob:"1997-07-09",age:29,h:178,w:88,isPitcher:true,wbcStats:{G:3,GS:1,W:0,L:0,H:0,SV:0,IP:3.0,SO:1,BB:2,HA:4,HR:0,ER:4,ERA:12.0,WHIP:2.0,Kpct:7.1,BBpct:14.3}},
  {id:73,name:"Duque Hebbert",team:"Nicaragua",pool:"Pool D",pos:"P",mlbId:808986,bats:"R",throws:"R",num:62,dob:"2001-10-29",age:25,h:178,w:77,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:2.0,SO:1,BB:2,HA:4,HR:0,ER:1,ERA:4.5,WHIP:3.0,Kpct:7.1,BBpct:14.3}},
  {id:74,name:"Elian Rayo",team:"Nicaragua",pool:"Pool D",pos:"3B",mlbId:692378,bats:"R",throws:"R",num:31,dob:"2003-03-04",age:23,h:183,w:92,isPitcher:false,wbcStats:{PA:4,AB:3,H:1,HR:0,R:0,RBI:0,BB:1,SO:1,SB:0,AVG:0.333,OBP:0.5}},
  {id:75,name:"Emanuel Trujillo",team:"Nicaragua",pool:"Pool D",pos:"1B",mlbId:693721,bats:"R",throws:"R",num:9,dob:"2001-10-19",age:25,h:185,w:93,isPitcher:false,wbcStats:{PA:20,AB:19,H:6,HR:1,R:2,RBI:2,BB:0,SO:6,SB:0,AVG:0.316,OBP:0.316}},
  {id:76,name:"Erasmo Ramirez",team:"Nicaragua",pool:"Pool D",pos:"P",mlbId:541640,bats:"R",throws:"R",num:30,dob:"1990-05-02",age:36,h:183,w:100,isPitcher:true,wbcStats:{G:2,GS:2,W:0,L:0,H:0,SV:0,IP:7.0,SO:3,BB:1,HA:7,HR:0,ER:2,ERA:2.57,WHIP:1.14,Kpct:10.0,BBpct:3.3}},
  {id:77,name:"Freddy Zamora",team:"Nicaragua",pool:"Pool D",pos:"SS",mlbId:668965,bats:"R",throws:"R",num:23,dob:"1998-11-01",age:28,h:183,w:84,isPitcher:false,wbcStats:{PA:15,AB:13,H:2,HR:1,R:2,RBI:3,BB:1,SO:4,SB:0,AVG:0.154,OBP:0.214}},
  {id:78,name:"Ismael Munguia",team:"Nicaragua",pool:"Pool D",pos:"OF",mlbId:665998,bats:"L",throws:"L",num:18,dob:"1998-10-19",age:28,h:173,w:72,isPitcher:false,wbcStats:{PA:24,AB:20,H:10,HR:0,R:1,RBI:1,BB:2,SO:2,SB:0,AVG:0.5,OBP:0.545}},
  {id:79,name:"JC Ramirez",team:"Nicaragua",pool:"Pool D",pos:"P",mlbId:500724,bats:"R",throws:"R",num:66,dob:"1988-08-16",age:38,h:191,w:79,isPitcher:true,wbcStats:{G:2,GS:0,W:1,L:0,H:0,SV:0,IP:1.2,SO:3,BB:1,HA:3,HR:0,ER:0,ERA:0.0,WHIP:2.4,Kpct:33.3,BBpct:11.1}},
  {id:80,name:"Jeter Downs",team:"Nicaragua",pool:"Pool D",pos:"2B",mlbId:669023,bats:"R",throws:"R",num:4,dob:"1998-07-27",age:28,h:178,w:89,isPitcher:false,wbcStats:{PA:16,AB:15,H:4,HR:1,R:1,RBI:2,BB:0,SO:5,SB:0,AVG:0.267,OBP:0.267}},
  {id:81,name:"Jose Orozco",team:"Nicaragua",pool:"Pool D",pos:"IF",mlbId:821829,bats:"L",throws:"R",num:20,dob:"1999-06-03",age:27,h:170,w:79,isPitcher:false,wbcStats:{PA:4,AB:4,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0,AVG:0.0,OBP:0.0}},
  {id:82,name:"Juan Montes",team:"Nicaragua",pool:"Pool D",pos:"OF",mlbId:650375,bats:"R",throws:"R",num:99,dob:"1995-05-15",age:31,h:185,w:91,isPitcher:false,wbcStats:{PA:6,AB:5,H:0,HR:0,R:0,RBI:0,BB:1,SO:3,SB:0,AVG:0.0,OBP:0.167}},
  {id:83,name:"Kenword Burton",team:"Nicaragua",pool:"Pool D",pos:"P",mlbId:830693,bats:"R",throws:"R",num:10,dob:"1998-08-04",age:28,h:183,w:89,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:3.0,SO:1,BB:2,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.67,Kpct:10.0,BBpct:20.0}},
  {id:84,name:"Mark Vientos",team:"Nicaragua",pool:"Pool D",pos:"3B",mlbId:668901,bats:"R",throws:"R",num:13,dob:"1999-12-11",age:27,h:191,w:84,isPitcher:false,wbcStats:{PA:23,AB:22,H:5,HR:0,R:0,RBI:0,BB:1,SO:9,SB:0,AVG:0.227,OBP:0.261}},
  {id:85,name:"Melvin Novoa",team:"Nicaragua",pool:"Pool D",pos:"C",mlbId:649700,bats:"R",throws:"R",num:75,dob:"1996-06-17",age:30,h:180,w:91,isPitcher:false,wbcStats:{PA:12,AB:11,H:2,HR:0,R:0,RBI:0,BB:1,SO:4,SB:0,AVG:0.182,OBP:0.25}},
  {id:86,name:"Omar Mendoza",team:"Nicaragua",pool:"Pool D",pos:"OF",mlbId:821830,bats:"R",throws:"R",num:45,dob:"1997-10-30",age:29,h:173,w:95,isPitcher:false,wbcStats:{PA:12,AB:11,H:2,HR:0,R:0,RBI:0,BB:1,SO:2,SB:0,AVG:0.182,OBP:0.25}},
  {id:87,name:"Oscar Rayo",team:"Nicaragua",pool:"Pool D",pos:"P",mlbId:699325,bats:"L",throws:"L",num:90,dob:"2002-01-03",age:24,h:185,w:82,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:2.1,SO:2,BB:4,HA:2,HR:0,ER:0,ERA:0.0,WHIP:2.57,Kpct:15.4,BBpct:30.8}},
  {id:88,name:"Osman Gutierrez",team:"Nicaragua",pool:"Pool D",pos:"P",mlbId:608476,bats:"R",throws:"R",num:41,dob:"1994-12-15",age:32,h:193,w:113,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:1.1,SO:1,BB:2,HA:4,HR:2,ER:5,ERA:33.75,WHIP:4.5,Kpct:10.0,BBpct:20.0}},
  {id:89,name:"Ronald Medrano",team:"Nicaragua",pool:"Pool D",pos:"P",mlbId:642574,bats:"R",throws:"R",num:36,dob:"1995-09-17",age:31,h:183,w:77,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:2.2,SO:3,BB:2,HA:6,HR:0,ER:3,ERA:10.12,WHIP:3.0,Kpct:18.8,BBpct:12.5}},
  {id:90,name:"Ronald Rivera",team:"Nicaragua",pool:"Pool D",pos:"C",mlbId:830696,bats:"R",throws:"R",num:25,dob:"1993-11-28",age:33,h:183,w:91,isPitcher:false,wbcStats:{PA:9,AB:8,H:1,HR:0,R:0,RBI:0,BB:1,SO:4,SB:0,AVG:0.125,OBP:0.222}},
  {id:91,name:"Stiven Cruz",team:"Nicaragua",pool:"Pool D",pos:"P",mlbId:691602,bats:"R",throws:"R",num:14,dob:"2001-11-14",age:25,h:188,w:75,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:1,H:0,SV:0,IP:4.1,SO:5,BB:0,HA:5,HR:2,ER:3,ERA:6.23,WHIP:1.15,Kpct:29.4,BBpct:0.0}},
  {id:92,name:"Alejandro Kirk",team:"Mexico",pool:"Pool B",pos:"C",mlbId:672386,bats:"R",throws:"R",num:30,dob:"1998-11-06",age:28,h:173,w:111,isPitcher:false,wbcStats:{PA:18,AB:18,H:7,HR:1,R:4,RBI:5,BB:0,SO:2,SB:0,AVG:0.389,OBP:0.389}},
  {id:93,name:"Alejandro Osuna",team:"Mexico",pool:"Pool B",pos:"OF",mlbId:696030,bats:"L",throws:"L",num:19,dob:"2002-10-10",age:24,h:175,w:84,isPitcher:false,wbcStats:{PA:5,AB:5,H:1,HR:1,R:1,RBI:1,BB:0,SO:2,SB:0,AVG:0.2,OBP:0.2}},
  {id:94,name:"Alek Thomas",team:"Mexico",pool:"Pool B",pos:"OF",mlbId:677950,bats:"L",throws:"L",num:5,dob:"2000-04-28",age:26,h:175,w:79,isPitcher:false,wbcStats:{PA:18,AB:17,H:6,HR:1,R:3,RBI:6,BB:1,SO:6,SB:0,AVG:0.353,OBP:0.389}},
  {id:95,name:"Alex Carrillo",team:"Mexico",pool:"Pool B",pos:"P",mlbId:692024,bats:"R",throws:"R",num:24,dob:"1997-06-06",age:29,h:188,w:100,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:2.2,SO:4,BB:3,HA:3,HR:1,ER:1,ERA:3.38,WHIP:2.25,Kpct:30.8,BBpct:23.1}},
  {id:96,name:"Alexander Armenta",team:"Mexico",pool:"Pool B",pos:"P",mlbId:694401,bats:"L",throws:"L",num:35,dob:"2004-06-26",age:22,h:175,w:85,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:2.0,SO:4,BB:0,HA:4,HR:0,ER:1,ERA:4.5,WHIP:2.0,Kpct:40.0,BBpct:0.0}},
  {id:97,name:"Alexis Wilson",team:"Mexico",pool:"Pool B",pos:"C",mlbId:656029,bats:"R",throws:"R",num:26,dob:"1996-08-13",age:30,h:178,w:91,isPitcher:false,wbcStats:{PA:3,AB:3,H:1,HR:0,R:0,RBI:1,BB:0,SO:0,SB:0,AVG:0.333,OBP:0.333}},
  {id:98,name:"Andres Munoz",team:"Mexico",pool:"Pool B",pos:"P",mlbId:662253,bats:"R",throws:"R",num:75,dob:"1999-01-16",age:27,h:188,w:101,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:1.0,SO:2,BB:0,HA:2,HR:0,ER:1,ERA:9.0,WHIP:2.0,Kpct:40.0,BBpct:0.0}},
  {id:99,name:"Brennan Bernardino",team:"Mexico",pool:"Pool B",pos:"P",mlbId:657514,bats:"L",throws:"L",num:22,dob:"1992-01-15",age:34,h:193,w:82,isPitcher:true,wbcStats:{G:2,GS:1,W:0,L:0,H:1,SV:0,IP:2.0,SO:2,BB:0,HA:2,HR:0,ER:0,ERA:0.0,WHIP:1.0,Kpct:28.6,BBpct:0.0}},
  {id:100,name:"Daniel Duarte",team:"Mexico",pool:"Pool B",pos:"P",mlbId:650960,bats:"R",throws:"R",num:53,dob:"1996-12-04",age:30,h:183,w:107,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:2.0,SO:3,BB:1,HA:1,HR:0,ER:0,ERA:0.0,WHIP:1.0,Kpct:37.5,BBpct:12.5}},
  {id:101,name:"Gerardo Reyes",team:"Mexico",pool:"Pool B",pos:"P",mlbId:622103,bats:"R",throws:"R",num:33,dob:"1993-05-13",age:33,h:180,w:88,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:2.0,SO:3,BB:0,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.5,Kpct:37.5,BBpct:0.0}},
  {id:102,name:"Jared Serna",team:"Mexico",pool:"Pool B",pos:"2B",mlbId:691858,bats:"R",throws:"R",num:59,dob:"2002-06-01",age:24,h:170,w:79,isPitcher:false,wbcStats:{PA:4,AB:3,H:1,HR:0,R:2,RBI:0,BB:0,SO:0,SB:0,AVG:0.333,OBP:0.333}},
  {id:103,name:"Jarren Duran",team:"Mexico",pool:"Pool B",pos:"OF",mlbId:680776,bats:"L",throws:"R",num:16,dob:"1996-09-05",age:30,h:183,w:93,isPitcher:false,wbcStats:{PA:19,AB:17,H:6,HR:3,R:6,RBI:5,BB:1,SO:5,SB:2,AVG:0.353,OBP:0.389}},
  {id:104,name:"Javier Assad",team:"Mexico",pool:"Pool B",pos:"P",mlbId:665871,bats:"R",throws:"R",num:77,dob:"1997-07-30",age:29,h:185,w:91,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:3.2,SO:2,BB:1,HA:2,HR:0,ER:0,ERA:0.0,WHIP:0.82,Kpct:15.4,BBpct:7.7}},
  {id:105,name:"Jesus Cruz",team:"Mexico",pool:"Pool B",pos:"P",mlbId:672911,bats:"R",throws:"R",num:49,dob:"1995-04-15",age:31,h:185,w:104,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:1,H:0,SV:0,IP:1.0,SO:2,BB:1,HA:7,HR:3,ER:7,ERA:63.0,WHIP:8.0,Kpct:15.4,BBpct:7.7}},
  {id:106,name:"Joey Meneses",team:"Mexico",pool:"Pool B",pos:"1B",mlbId:608841,bats:"R",throws:"R",num:32,dob:"1992-05-06",age:34,h:193,w:107,isPitcher:false,wbcStats:{PA:6,AB:6,H:3,HR:0,R:2,RBI:1,BB:0,SO:0,SB:0,AVG:0.5,OBP:0.5}},
  {id:107,name:"Joey Ortiz",team:"Mexico",pool:"Pool B",pos:"3B",mlbId:687401,bats:"R",throws:"R",num:7,dob:"1998-07-14",age:28,h:178,w:87,isPitcher:false,wbcStats:{PA:17,AB:16,H:3,HR:0,R:2,RBI:1,BB:1,SO:4,SB:0,AVG:0.188,OBP:0.235}},
  {id:108,name:"Jonathan Aranda",team:"Mexico",pool:"Pool B",pos:"2B",mlbId:666018,bats:"L",throws:"R",num:8,dob:"1998-05-23",age:28,h:183,w:98,isPitcher:false,wbcStats:{PA:19,AB:16,H:7,HR:1,R:6,RBI:4,BB:3,SO:3,SB:0,AVG:0.438,OBP:0.526}},
  {id:109,name:"Julian Ornelas",team:"Mexico",pool:"Pool B",pos:"OF",mlbId:676140,bats:"L",throws:"R",num:31,dob:"1996-12-28",age:30,h:183,w:83,isPitcher:false,wbcStats:{PA:9,AB:7,H:1,HR:1,R:1,RBI:2,BB:2,SO:2,SB:0,AVG:0.143,OBP:0.333}},
  {id:110,name:"Luis Gastelum",team:"Mexico",pool:"Pool B",pos:"P",mlbId:703725,bats:"R",throws:"R",num:27,dob:"2001-09-27",age:25,h:188,w:79,isPitcher:true,wbcStats:{G:3,GS:0,W:1,L:0,H:1,SV:0,IP:2.2,SO:1,BB:0,HA:3,HR:0,ER:2,ERA:6.75,WHIP:1.12,Kpct:9.1,BBpct:0.0}},
  {id:111,name:"Manny Barreda",team:"Mexico",pool:"Pool B",pos:"P",mlbId:518435,bats:"R",throws:"R",num:50,dob:"1988-10-08",age:38,h:180,w:88,isPitcher:true,wbcStats:{G:2,GS:1,W:0,L:0,H:0,SV:0,IP:3.1,SO:3,BB:4,HA:5,HR:0,ER:2,ERA:5.4,WHIP:2.7,Kpct:16.7,BBpct:22.2}},
  {id:112,name:"Mateo Gil",team:"Mexico",pool:"Pool B",pos:"3B",mlbId:680771,bats:"R",throws:"R",num:3,dob:"2000-07-24",age:26,h:183,w:82,isPitcher:false,wbcStats:{PA:2,AB:2,H:0,HR:0,R:0,RBI:0,BB:0,SO:1,SB:0,AVG:0.0,OBP:0.0}},
  {id:113,name:"Nacho Alvarez Jr.",team:"Mexico",pool:"Pool B",pos:"SS",mlbId:805373,bats:"R",throws:"R",num:2,dob:"2003-04-11",age:23,h:178,w:86,isPitcher:false,wbcStats:{PA:15,AB:14,H:6,HR:1,R:3,RBI:4,BB:1,SO:1,SB:1,AVG:0.429,OBP:0.467}},
  {id:114,name:"Nick Gonzales",team:"Mexico",pool:"Pool B",pos:"2B",mlbId:693304,bats:"R",throws:"R",num:13,dob:"1999-05-27",age:27,h:173,w:88,isPitcher:false,wbcStats:{PA:18,AB:17,H:3,HR:0,R:3,RBI:3,BB:1,SO:4,SB:1,AVG:0.176,OBP:0.222}},
  {id:115,name:"Randy Arozarena",team:"Mexico",pool:"Pool B",pos:"OF",mlbId:668227,bats:"R",throws:"R",num:56,dob:"1995-02-28",age:31,h:178,w:84,isPitcher:false,wbcStats:{PA:15,AB:12,H:2,HR:0,R:3,RBI:1,BB:2,SO:2,SB:0,AVG:0.167,OBP:0.286}},
  {id:116,name:"Robert Garcia",team:"Mexico",pool:"Pool B",pos:"P",mlbId:676395,bats:"R",throws:"L",num:62,dob:"1996-06-14",age:30,h:193,w:107,isPitcher:true,wbcStats:{G:3,GS:0,W:1,L:0,H:1,SV:0,IP:3.0,SO:2,BB:2,HA:2,HR:0,ER:0,ERA:0.0,WHIP:1.33,Kpct:15.4,BBpct:15.4}},
  {id:117,name:"Roel Ramirez",team:"Mexico",pool:"Pool B",pos:"P",mlbId:641995,bats:"R",throws:"R",num:55,dob:"1995-05-26",age:31,h:183,w:107,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:1,IP:3.2,SO:4,BB:2,HA:4,HR:0,ER:0,ERA:0.0,WHIP:1.64,Kpct:23.5,BBpct:11.8}},
  {id:118,name:"Rowdy Tellez",team:"Mexico",pool:"Pool B",pos:"1B",mlbId:642133,bats:"L",throws:"L",num:44,dob:"1995-03-16",age:31,h:193,w:122,isPitcher:false,wbcStats:{PA:17,AB:14,H:3,HR:1,R:2,RBI:2,BB:3,SO:5,SB:0,AVG:0.214,OBP:0.353}},
  {id:119,name:"Samy Natera Jr.",team:"Mexico",pool:"Pool B",pos:"P",mlbId:696519,bats:"L",throws:"L",num:17,dob:"1999-11-05",age:27,h:193,w:104,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:2.1,SO:2,BB:4,HA:2,HR:0,ER:0,ERA:0.0,WHIP:2.57,Kpct:16.7,BBpct:33.3}},
  {id:120,name:"Taijuan Walker",team:"Mexico",pool:"Pool B",pos:"P",mlbId:592836,bats:"R",throws:"R",num:99,dob:"1992-08-13",age:34,h:193,w:107,isPitcher:true,wbcStats:{G:2,GS:2,W:1,L:0,H:0,SV:0,IP:5.2,SO:5,BB:4,HA:2,HR:0,ER:0,ERA:0.0,WHIP:1.06,Kpct:21.7,BBpct:17.4}},
  {id:121,name:"Victor Vodnik",team:"Mexico",pool:"Pool B",pos:"P",mlbId:680767,bats:"R",throws:"R",num:11,dob:"1999-10-09",age:27,h:183,w:91,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:1,SV:0,IP:1.0,SO:0,BB:2,HA:0,HR:0,ER:0,ERA:0.0,WHIP:2.0,Kpct:0.0,BBpct:40.0}},
  {id:122,name:"Abraham Toro",team:"Canada",pool:"Pool A",pos:"3B",mlbId:647351,bats:"S",throws:"R",num:31,dob:"1996-12-20",age:30,h:183,w:101,isPitcher:false,wbcStats:{PA:20,AB:17,H:6,HR:0,R:4,RBI:7,BB:3,SO:2,SB:0,AVG:0.353,OBP:0.45}},
  {id:123,name:"Adam Hall",team:"Canada",pool:"Pool A",pos:"SS",mlbId:647228,bats:"R",throws:"R",num:2,dob:"1999-05-22",age:27,h:180,w:75,isPitcher:false,wbcStats:{PA:2,AB:2,H:0,HR:0,R:0,RBI:0,BB:0,SO:1,SB:0,AVG:0.0,OBP:0.0}},
  {id:124,name:"Adam Macko",team:"Canada",pool:"Pool A",pos:"P",mlbId:671936,bats:"L",throws:"L",num:64,dob:"2000-12-30",age:26,h:183,w:77,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.1,SO:1,BB:0,HA:2,HR:0,ER:0,ERA:0.0,WHIP:1.5,Kpct:20.0,BBpct:0.0}},
  {id:125,name:"Antoine Jean",team:"Canada",pool:"Pool A",pos:"P",mlbId:683328,bats:"R",throws:"L",num:14,dob:"2001-08-01",age:25,h:188,w:86,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:2.1,SO:3,BB:1,HA:2,HR:0,ER:0,ERA:0.0,WHIP:1.29,Kpct:30.0,BBpct:10.0}},
  {id:126,name:"Bo Naylor",team:"Canada",pool:"Pool A",pos:"C",mlbId:666310,bats:"L",throws:"R",num:23,dob:"2000-02-21",age:26,h:175,w:93,isPitcher:false,wbcStats:{PA:12,AB:11,H:3,HR:0,R:4,RBI:1,BB:1,SO:1,SB:0,AVG:0.273,OBP:0.333}},
  {id:127,name:"Brock Dykxhoorn",team:"Canada",pool:"Pool A",pos:"P",mlbId:621245,bats:"R",throws:"R",num:44,dob:"1994-07-02",age:32,h:203,w:113,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:1,H:0,SV:1,IP:4.0,SO:1,BB:1,HA:3,HR:1,ER:3,ERA:6.75,WHIP:1.0,Kpct:6.2,BBpct:6.2}},
  {id:128,name:"Cal Quantrill",team:"Canada",pool:"Pool A",pos:"P",mlbId:615698,bats:"L",throws:"R",num:47,dob:"1995-02-10",age:31,h:191,w:88,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:2.0,SO:0,BB:0,HA:2,HR:0,ER:1,ERA:4.5,WHIP:1.0,Kpct:0.0,BBpct:0.0}},
  {id:129,name:"Denzel Clarke",team:"Canada",pool:"Pool A",pos:"OF",mlbId:672016,bats:"R",throws:"R",num:1,dob:"2000-05-01",age:26,h:191,w:100,isPitcher:false,wbcStats:{PA:17,AB:14,H:3,HR:0,R:3,RBI:3,BB:1,SO:4,SB:0,AVG:0.214,OBP:0.267}},
  {id:130,name:"Edouard Julien",team:"Canada",pool:"Pool A",pos:"2B",mlbId:666397,bats:"L",throws:"R",num:15,dob:"1999-04-30",age:27,h:185,w:88,isPitcher:false,wbcStats:{PA:18,AB:16,H:2,HR:0,R:2,RBI:0,BB:2,SO:7,SB:0,AVG:0.125,OBP:0.222}},
  {id:131,name:"Eric Cerantola",team:"Canada",pool:"Pool A",pos:"P",mlbId:672021,bats:"R",throws:"R",num:61,dob:"2000-05-02",age:26,h:196,w:102,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:2.0,SO:3,BB:0,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.0,Kpct:50.0,BBpct:0.0}},
  {id:132,name:"Indigo Diaz",team:"Canada",pool:"Pool A",pos:"P",mlbId:666301,bats:"R",throws:"R",num:52,dob:"1998-10-14",age:28,h:196,w:113,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:2.0,SO:3,BB:1,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.5,Kpct:42.9,BBpct:14.3}},
  {id:133,name:"Jacob Robson",team:"Canada",pool:"Pool A",pos:"OF",mlbId:615699,bats:"L",throws:"R",num:8,dob:"1994-11-20",age:32,h:178,w:83,isPitcher:false,wbcStats:{PA:4,AB:4,H:2,HR:1,R:1,RBI:2,BB:0,SO:2,SB:0,AVG:0.5,OBP:0.5}},
  {id:134,name:"James Paxton",team:"Canada",pool:"Pool A",pos:"P",mlbId:572020,bats:"L",throws:"L",num:65,dob:"1988-11-06",age:38,h:193,w:96,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:1,H:0,SV:0,IP:2.2,SO:2,BB:2,HA:6,HR:1,ER:1,ERA:3.38,WHIP:3.0,Kpct:11.8,BBpct:11.8}},
  {id:135,name:"Jameson Taillon",team:"Canada",pool:"Pool A",pos:"P",mlbId:592791,bats:"R",throws:"R",num:50,dob:"1991-11-18",age:35,h:196,w:104,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:3.2,SO:3,BB:2,HA:2,HR:0,ER:1,ERA:2.45,WHIP:1.09,Kpct:21.4,BBpct:14.3}},
  {id:136,name:"Jared Young",team:"Canada",pool:"Pool A",pos:"OF",mlbId:676724,bats:"L",throws:"R",num:29,dob:"1995-07-09",age:31,h:188,w:91,isPitcher:false,wbcStats:{PA:11,AB:10,H:3,HR:0,R:0,RBI:1,BB:0,SO:3,SB:0,AVG:0.3,OBP:0.3}},
  {id:137,name:"Jordan Balazovic",team:"Canada",pool:"Pool A",pos:"P",mlbId:666364,bats:"R",throws:"R",num:16,dob:"1998-09-17",age:28,h:196,w:98,isPitcher:true,wbcStats:{G:2,GS:1,W:1,L:0,H:0,SV:0,IP:4.0,SO:4,BB:3,HA:2,HR:0,ER:1,ERA:2.25,WHIP:1.25,Kpct:28.6,BBpct:21.4}},
  {id:138,name:"Josh Naylor",team:"Canada",pool:"Pool A",pos:"1B",mlbId:647304,bats:"L",throws:"L",num:12,dob:"1997-06-22",age:29,h:178,w:107,isPitcher:false,wbcStats:{PA:20,AB:16,H:4,HR:0,R:2,RBI:1,BB:2,SO:1,SB:2,AVG:0.25,OBP:0.333}},
  {id:139,name:"Liam Hicks",team:"Canada",pool:"Pool A",pos:"C",mlbId:689414,bats:"L",throws:"R",num:34,dob:"1999-06-02",age:27,h:175,w:84,isPitcher:false,wbcStats:{PA:8,AB:6,H:2,HR:0,R:2,RBI:1,BB:1,SO:1,SB:0,AVG:0.333,OBP:0.429}},
  {id:140,name:"Logan Allen",team:"Canada",pool:"Pool A",pos:"P",mlbId:663531,bats:"R",throws:"L",num:22,dob:"1997-05-23",age:29,h:191,w:91,isPitcher:true,wbcStats:{G:3,GS:1,W:0,L:0,H:1,SV:0,IP:4.1,SO:2,BB:5,HA:6,HR:0,ER:6,ERA:12.46,WHIP:2.54,Kpct:8.7,BBpct:21.7}},
  {id:141,name:"Matt Davidson",team:"Canada",pool:"Pool A",pos:"3B",mlbId:571602,bats:"R",throws:"R",num:24,dob:"1991-03-26",age:35,h:191,w:104,isPitcher:false,wbcStats:{PA:8,AB:7,H:2,HR:0,R:2,RBI:0,BB:0,SO:3,SB:0,AVG:0.286,OBP:0.286}},
  {id:142,name:"Matt Wilkinson",team:"Canada",pool:"Pool A",pos:"P",mlbId:683363,bats:"R",throws:"L",num:35,dob:"2002-12-10",age:24,h:185,w:113,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:1,IP:2.2,SO:7,BB:0,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.38,Kpct:77.8,BBpct:0.0}},
  {id:143,name:"Micah Ashman",team:"Canada",pool:"Pool A",pos:"P",mlbId:809840,bats:"L",throws:"L",num:57,dob:"2002-08-22",age:24,h:201,w:88,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:2,SV:0,IP:1.2,SO:2,BB:2,HA:1,HR:0,ER:1,ERA:5.4,WHIP:1.8,Kpct:22.2,BBpct:22.2}},
  {id:144,name:"Michael Soroka",team:"Canada",pool:"Pool A",pos:"P",mlbId:647336,bats:"R",throws:"R",num:40,dob:"1997-08-04",age:29,h:196,w:113,isPitcher:true,wbcStats:{G:1,GS:1,W:1,L:0,H:0,SV:0,IP:3.0,SO:2,BB:1,HA:4,HR:0,ER:1,ERA:3.0,WHIP:1.67,Kpct:16.7,BBpct:8.3}},
  {id:145,name:"Noah Skirrow",team:"Canada",pool:"Pool A",pos:"P",mlbId:659418,bats:"R",throws:"R",num:25,dob:"1998-07-21",age:28,h:191,w:98,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:5.0,SO:6,BB:2,HA:3,HR:0,ER:1,ERA:1.8,WHIP:1.0,Kpct:30.0,BBpct:10.0}},
  {id:146,name:"Otto Lopez",team:"Canada",pool:"Pool A",pos:"SS",mlbId:672640,bats:"R",throws:"R",num:6,dob:"1998-10-01",age:28,h:178,w:84,isPitcher:false,wbcStats:{PA:19,AB:16,H:2,HR:0,R:2,RBI:0,BB:1,SO:4,SB:0,AVG:0.125,OBP:0.176}},
  {id:147,name:"Owen Caissie",team:"Canada",pool:"Pool A",pos:"OF",mlbId:683357,bats:"L",throws:"R",num:21,dob:"2002-07-08",age:24,h:193,w:86,isPitcher:false,wbcStats:{PA:16,AB:15,H:7,HR:1,R:3,RBI:4,BB:1,SO:6,SB:0,AVG:0.467,OBP:0.5}},
  {id:148,name:"Phillippe Aumont",team:"Canada",pool:"Pool A",pos:"P",mlbId:518418,bats:"L",throws:"R",num:37,dob:"1989-01-07",age:37,h:201,w:120,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:1.1,SO:1,BB:0,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.0,Kpct:25.0,BBpct:0.0}},
  {id:149,name:"Rob Zastryzny",team:"Canada",pool:"Pool A",pos:"P",mlbId:642239,bats:"R",throws:"L",num:58,dob:"1992-03-26",age:34,h:191,w:105,isPitcher:true,wbcStats:{G:2,GS:0,W:1,L:0,H:0,SV:0,IP:2.0,SO:0,BB:0,HA:3,HR:0,ER:0,ERA:0.0,WHIP:1.5,Kpct:0.0,BBpct:0.0}},
  {id:150,name:"Tyler Black",team:"Canada",pool:"Pool A",pos:"2B",mlbId:672012,bats:"L",throws:"R",num:7,dob:"2000-07-26",age:26,h:183,w:94,isPitcher:false,wbcStats:{PA:17,AB:12,H:3,HR:0,R:1,RBI:3,BB:3,SO:4,SB:3,AVG:0.25,OBP:0.4}},
  {id:151,name:"Tyler O'Neill",team:"Canada",pool:"Pool A",pos:"OF",mlbId:641933,bats:"R",throws:"R",num:9,dob:"1995-06-22",age:31,h:175,w:91,isPitcher:false,wbcStats:{PA:19,AB:13,H:1,HR:0,R:0,RBI:3,BB:5,SO:4,SB:1,AVG:0.077,OBP:0.333}},
  {id:152,name:"Aaron Nola",team:"Italy",pool:"Pool B",pos:"P",mlbId:605400,bats:"R",throws:"R",num:27,dob:"1993-06-04",age:33,h:188,w:91,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:3.0,SO:4,BB:0,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.33,Kpct:40.0,BBpct:0.0}},
  {id:153,name:"Adam Ottavino",team:"Italy",pool:"Pool B",pos:"P",mlbId:493603,bats:"S",throws:"R",num:0,dob:"1985-11-22",age:41,h:196,w:112,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:1,SV:0,IP:1.0,SO:1,BB:1,HA:1,HR:0,ER:1,ERA:9.0,WHIP:2.0,Kpct:20.0,BBpct:20.0}},
  {id:154,name:"Alek Jacob",team:"Italy",pool:"Pool B",pos:"P",mlbId:689690,bats:"L",throws:"R",num:37,dob:"1998-06-16",age:28,h:191,w:86,isPitcher:true,wbcStats:{G:3,GS:0,W:1,L:0,H:1,SV:0,IP:2.2,SO:3,BB:0,HA:3,HR:2,ER:3,ERA:10.12,WHIP:1.12,Kpct:27.3,BBpct:0.0}},
  {id:155,name:"Andrew Fischer",team:"Italy",pool:"Pool B",pos:"3B",mlbId:702652,bats:"L",throws:"R",num:11,dob:"2004-05-25",age:22,h:183,w:95,isPitcher:false,wbcStats:{PA:7,AB:6,H:2,HR:1,R:2,RBI:2,BB:1,SO:2,SB:0,AVG:0.333,OBP:0.429}},
  {id:156,name:"Claudio Scotti",team:"Italy",pool:"Pool B",pos:"P",mlbId:668144,bats:"R",throws:"R",num:98,dob:"1998-07-08",age:28,h:193,w:95,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:4.0,SO:1,BB:1,HA:2,HR:0,ER:0,ERA:0.0,WHIP:0.75,Kpct:6.7,BBpct:6.7}},
  {id:157,name:"Dan Altavilla",team:"Italy",pool:"Pool B",pos:"P",mlbId:656186,bats:"R",throws:"R",num:53,dob:"1992-09-08",age:34,h:180,w:107,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:2.0,SO:2,BB:2,HA:1,HR:1,ER:1,ERA:4.5,WHIP:1.5,Kpct:22.2,BBpct:22.2}},
  {id:158,name:"Dante Nori",team:"Italy",pool:"Pool B",pos:"OF",mlbId:807276,bats:"L",throws:"L",num:16,dob:"2004-10-07",age:22,h:175,w:86,isPitcher:false,wbcStats:{PA:19,AB:16,H:8,HR:2,R:4,RBI:5,BB:2,SO:1,SB:1,AVG:0.5,OBP:0.556}},
  {id:159,name:"Dominic Canzone",team:"Italy",pool:"Pool B",pos:"RF",mlbId:686527,bats:"L",throws:"R",num:8,dob:"1997-08-16",age:29,h:183,w:86,isPitcher:false,wbcStats:{PA:22,AB:20,H:2,HR:1,R:2,RBI:4,BB:2,SO:6,SB:0,AVG:0.1,OBP:0.182}},
  {id:160,name:"Dylan DeLucia",team:"Italy",pool:"Pool B",pos:"P",mlbId:690626,bats:"R",throws:"R",num:22,dob:"2000-08-01",age:26,h:180,w:100,isPitcher:true,wbcStats:{G:2,GS:2,W:0,L:0,H:0,SV:0,IP:5.1,SO:7,BB:3,HA:3,HR:1,ER:3,ERA:5.06,WHIP:1.12,Kpct:30.4,BBpct:13.0}},
  {id:161,name:"Gabriele Quattrini",team:"Italy",pool:"Pool B",pos:"P",mlbId:838341,bats:"R",throws:"R",num:80,dob:"1996-07-18",age:30,h:191,w:118,isPitcher:true,wbcStats:{G:2,GS:1,W:1,L:0,H:0,SV:0,IP:4.1,SO:4,BB:2,HA:2,HR:0,ER:0,ERA:0.0,WHIP:0.92,Kpct:23.5,BBpct:11.8}},
  {id:162,name:"Gordon Graceffo",team:"Italy",pool:"Pool B",pos:"P",mlbId:700669,bats:"R",throws:"R",num:44,dob:"2000-03-17",age:26,h:193,w:100,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:2,SV:0,IP:2.0,SO:4,BB:1,HA:2,HR:0,ER:0,ERA:0.0,WHIP:1.5,Kpct:44.4,BBpct:11.1}},
  {id:163,name:"Greg Weissert",team:"Italy",pool:"Pool B",pos:"P",mlbId:669711,bats:"R",throws:"R",num:57,dob:"1995-02-04",age:31,h:188,w:107,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:2,IP:2.2,SO:4,BB:2,HA:3,HR:0,ER:0,ERA:0.0,WHIP:1.88,Kpct:30.8,BBpct:15.4}},
  {id:164,name:"J.J. D'Orazio",team:"Italy",pool:"Pool B",pos:"C",mlbId:683490,bats:"R",throws:"R",num:28,dob:"2001-12-28",age:25,h:185,w:77,isPitcher:false,wbcStats:{PA:8,AB:7,H:3,HR:1,R:4,RBI:1,BB:1,SO:2,SB:0,AVG:0.429,OBP:0.5}},
  {id:165,name:"Jac Caglianone",team:"Italy",pool:"Pool B",pos:"RF",mlbId:695506,bats:"L",throws:"L",num:14,dob:"2003-02-09",age:23,h:193,w:113,isPitcher:false,wbcStats:{PA:14,AB:11,H:5,HR:1,R:4,RBI:3,BB:2,SO:1,SB:0,AVG:0.455,OBP:0.538}},
  {id:166,name:"Jakob Marsee",team:"Italy",pool:"Pool B",pos:"CF",mlbId:805300,bats:"L",throws:"L",num:5,dob:"2001-06-28",age:25,h:183,w:82,isPitcher:false,wbcStats:{PA:21,AB:18,H:4,HR:0,R:1,RBI:0,BB:3,SO:5,SB:1,AVG:0.222,OBP:0.333}},
  {id:167,name:"Joe La Sorsa",team:"Italy",pool:"Pool B",pos:"P",mlbId:686747,bats:"L",throws:"L",num:75,dob:"1998-04-29",age:28,h:196,w:102,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:3.0,SO:3,BB:0,HA:2,HR:0,ER:1,ERA:3.0,WHIP:0.67,Kpct:27.3,BBpct:0.0}},
  {id:168,name:"Jon Berti",team:"Italy",pool:"Pool B",pos:"3B",mlbId:542932,bats:"R",throws:"R",num:1,dob:"1990-01-22",age:36,h:178,w:86,isPitcher:false,wbcStats:{PA:17,AB:16,H:4,HR:1,R:3,RBI:2,BB:1,SO:5,SB:1,AVG:0.25,OBP:0.294}},
  {id:169,name:"Kyle Nicolas",team:"Italy",pool:"Pool B",pos:"P",mlbId:693312,bats:"R",throws:"R",num:19,dob:"1999-02-22",age:27,h:191,w:98,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:1.1,SO:1,BB:1,HA:2,HR:0,ER:2,ERA:13.5,WHIP:2.25,Kpct:12.5,BBpct:12.5}},
  {id:170,name:"Kyle Teel",team:"Italy",pool:"Pool B",pos:"C",mlbId:691019,bats:"L",throws:"R",num:3,dob:"2002-02-15",age:24,h:180,w:95,isPitcher:false,wbcStats:{PA:13,AB:12,H:6,HR:2,R:3,RBI:3,BB:1,SO:3,SB:0,AVG:0.5,OBP:0.538}},
  {id:171,name:"Matt Festa",team:"Italy",pool:"Pool B",pos:"P",mlbId:670036,bats:"R",throws:"R",num:52,dob:"1993-03-11",age:33,h:185,w:88,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.2,SO:3,BB:0,HA:4,HR:0,ER:1,ERA:5.4,WHIP:2.4,Kpct:33.3,BBpct:0.0}},
  {id:172,name:"Michael Lorenzen",team:"Italy",pool:"Pool B",pos:"P",mlbId:547179,bats:"R",throws:"R",num:24,dob:"1992-01-04",age:34,h:185,w:98,isPitcher:true,wbcStats:{G:1,GS:1,W:1,L:0,H:0,SV:0,IP:4.2,SO:2,BB:1,HA:2,HR:0,ER:0,ERA:0.0,WHIP:0.64,Kpct:11.8,BBpct:5.9}},
  {id:173,name:"Miles Mastrobuoni",team:"Italy",pool:"Pool B",pos:"3B",mlbId:670156,bats:"L",throws:"R",num:2,dob:"1995-10-31",age:31,h:175,w:84,isPitcher:false,wbcStats:{PA:9,AB:7,H:2,HR:0,R:0,RBI:1,BB:2,SO:3,SB:0,AVG:0.286,OBP:0.444}},
  {id:174,name:"Nick Morabito",team:"Italy",pool:"Pool B",pos:"OF",mlbId:703492,bats:"R",throws:"R",num:7,dob:"2003-05-07",age:23,h:178,w:84,isPitcher:false,wbcStats:{PA:3,AB:1,H:1,HR:0,R:1,RBI:0,BB:2,SO:0,SB:1,AVG:1.0,OBP:1.0}},
  {id:175,name:"Renzo Martini",team:"Italy",pool:"Pool B",pos:"3B",mlbId:606331,bats:"R",throws:"R",num:41,dob:"1992-08-25",age:34,h:185,w:86,isPitcher:false,wbcStats:{PA:3,AB:3,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0,AVG:0.0,OBP:0.0}},
  {id:176,name:"Ron Marinaccio",team:"Italy",pool:"Pool B",pos:"P",mlbId:676760,bats:"R",throws:"R",num:97,dob:"1995-07-01",age:31,h:188,w:93,isPitcher:true,wbcStats:{G:3,GS:0,W:1,L:0,H:2,SV:0,IP:2.1,SO:4,BB:0,HA:3,HR:1,ER:1,ERA:3.86,WHIP:1.29,Kpct:40.0,BBpct:0.0}},
  {id:177,name:"Sam Aldegheri",team:"Italy",pool:"Pool B",pos:"P",mlbId:691951,bats:"L",throws:"L",num:12,dob:"2001-09-19",age:25,h:185,w:95,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:4.2,SO:8,BB:2,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.64,Kpct:53.3,BBpct:13.3}},
  {id:178,name:"Sam Antonacci",team:"Italy",pool:"Pool B",pos:"SS",mlbId:803011,bats:"L",throws:"R",num:10,dob:"2003-02-06",age:23,h:180,w:88,isPitcher:false,wbcStats:{PA:10,AB:8,H:3,HR:1,R:3,RBI:4,BB:0,SO:1,SB:0,AVG:0.375,OBP:0.375}},
  {id:179,name:"Thomas Saggese",team:"Italy",pool:"Pool B",pos:"2B",mlbId:695336,bats:"R",throws:"R",num:6,dob:"2002-04-10",age:24,h:178,w:84,isPitcher:false,wbcStats:{PA:8,AB:8,H:2,HR:1,R:1,RBI:1,BB:0,SO:3,SB:0,AVG:0.25,OBP:0.25}},
  {id:180,name:"Vinnie Pasquantino",team:"Italy",pool:"Pool B",pos:"1B",mlbId:686469,bats:"L",throws:"L",num:9,dob:"1997-10-10",age:29,h:191,w:111,isPitcher:false,wbcStats:{PA:21,AB:18,H:1,HR:0,R:1,RBI:1,BB:3,SO:2,SB:0,AVG:0.056,OBP:0.19}},
  {id:181,name:"Zach Dezenzo",team:"Italy",pool:"Pool B",pos:"IF",mlbId:701305,bats:"R",throws:"R",num:4,dob:"2000-05-11",age:26,h:196,w:100,isPitcher:false,wbcStats:{PA:18,AB:16,H:2,HR:0,R:3,RBI:1,BB:2,SO:5,SB:0,AVG:0.125,OBP:0.222}},
  {id:182,name:"Assaf Lowengart",team:"Israel",pool:"Pool D",pos:"OF",mlbId:808953,bats:"R",throws:"R",num:24,dob:"1998-03-01",age:28,h:185,w:88,isPitcher:false,wbcStats:{PA:2,AB:2,H:0,HR:0,R:0,RBI:0,BB:0,SO:1,SB:0,AVG:0.0,OBP:0.0}},
  {id:183,name:"Ben Simon",team:"Israel",pool:"Pool D",pos:"P",mlbId:815084,bats:"R",throws:"R",num:15,dob:"2002-03-22",age:24,h:180,w:89,isPitcher:true,wbcStats:{G:3,GS:2,W:0,L:1,H:0,SV:0,IP:2.1,SO:4,BB:5,HA:3,HR:1,ER:4,ERA:15.43,WHIP:3.43,Kpct:26.7,BBpct:33.3}},
  {id:184,name:"Benjamin Rosengard",team:"Israel",pool:"Pool D",pos:"SS",mlbId:809780,bats:"L",throws:"R",num:22,dob:"2000-01-18",age:26,h:191,w:82,isPitcher:false,wbcStats:{PA:4,AB:3,H:1,HR:0,R:0,RBI:0,BB:1,SO:2,SB:0,AVG:0.333,OBP:0.5}},
  {id:185,name:"C.J. Stubbs",team:"Israel",pool:"Pool D",pos:"C",mlbId:667690,bats:"R",throws:"R",num:36,dob:"1996-11-12",age:30,h:188,w:94,isPitcher:false,wbcStats:{PA:15,AB:14,H:1,HR:0,R:0,RBI:0,BB:0,SO:6,SB:0,AVG:0.071,OBP:0.071}},
  {id:186,name:"Carlos Lequerica",team:"Israel",pool:"Pool D",pos:"P",mlbId:701557,bats:"R",throws:"R",num:40,dob:"2000-09-06",age:26,h:185,w:90,isPitcher:true,wbcStats:{G:2,GS:1,W:0,L:0,H:1,SV:0,IP:2.0,SO:1,BB:1,HA:2,HR:0,ER:2,ERA:9.0,WHIP:1.5,Kpct:10.0,BBpct:10.0}},
  {id:187,name:"Charlie Beilenson",team:"Israel",pool:"Pool D",pos:"P",mlbId:810897,bats:"R",throws:"R",num:5,dob:"1999-12-10",age:27,h:183,w:98,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:4.1,SO:2,BB:4,HA:1,HR:0,ER:0,ERA:0.0,WHIP:1.15,Kpct:11.8,BBpct:23.5}},
  {id:188,name:"Colby Halter",team:"Israel",pool:"Pool D",pos:"2B",mlbId:690985,bats:"L",throws:"R",num:7,dob:"2001-08-24",age:25,h:185,w:93,isPitcher:false,wbcStats:{PA:7,AB:7,H:0,HR:0,R:0,RBI:0,BB:0,SO:5,SB:0,AVG:0.0,OBP:0.0}},
  {id:189,name:"Cole Carrigg",team:"Israel",pool:"Pool D",pos:"SS",mlbId:694249,bats:"S",throws:"R",num:8,dob:"2002-05-08",age:24,h:188,w:91,isPitcher:false,wbcStats:{PA:22,AB:20,H:4,HR:0,R:4,RBI:1,BB:2,SO:3,SB:5,AVG:0.2,OBP:0.273}},
  {id:190,name:"Daniel Federman",team:"Israel",pool:"Pool D",pos:"P",mlbId:679946,bats:"L",throws:"R",num:99,dob:"1998-09-18",age:28,h:185,w:93,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:1,IP:3.1,SO:4,BB:4,HA:2,HR:0,ER:4,ERA:10.8,WHIP:1.8,Kpct:26.7,BBpct:26.7}},
  {id:191,name:"Dean Kremer",team:"Israel",pool:"Pool D",pos:"P",mlbId:665152,bats:"R",throws:"R",num:64,dob:"1996-01-07",age:30,h:188,w:95,isPitcher:true,wbcStats:{G:2,GS:1,W:1,L:0,H:0,SV:0,IP:8.1,SO:6,BB:2,HA:4,HR:0,ER:0,ERA:0.0,WHIP:0.72,Kpct:19.4,BBpct:6.5}},
  {id:192,name:"Garrett Stubbs",team:"Israel",pool:"Pool D",pos:"C",mlbId:596117,bats:"L",throws:"R",num:21,dob:"1993-05-26",age:33,h:175,w:77,isPitcher:false,wbcStats:{PA:22,AB:20,H:2,HR:0,R:1,RBI:1,BB:1,SO:5,SB:1,AVG:0.1,OBP:0.143}},
  {id:193,name:"Harrison Bader",team:"Israel",pool:"Pool D",pos:"OF",mlbId:664056,bats:"R",throws:"R",num:2,dob:"1994-06-03",age:32,h:180,w:95,isPitcher:false,wbcStats:{PA:14,AB:14,H:3,HR:1,R:2,RBI:2,BB:0,SO:5,SB:0,AVG:0.214,OBP:0.214}},
  {id:194,name:"Harrison Cohen",team:"Israel",pool:"Pool D",pos:"P",mlbId:694660,bats:"R",throws:"R",num:18,dob:"1999-05-28",age:27,h:183,w:96,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:3.0,SO:5,BB:0,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.0,Kpct:55.6,BBpct:0.0}},
  {id:195,name:"Jake Fishman",team:"Israel",pool:"Pool D",pos:"P",mlbId:670288,bats:"L",throws:"L",num:17,dob:"1995-02-08",age:31,h:191,w:88,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.2,SO:0,BB:0,HA:2,HR:0,ER:1,ERA:5.4,WHIP:1.2,Kpct:0.0,BBpct:0.0}},
  {id:196,name:"Jake Gelof",team:"Israel",pool:"Pool D",pos:"3B",mlbId:695372,bats:"R",throws:"R",num:45,dob:"2002-02-25",age:24,h:183,w:88,isPitcher:false,wbcStats:{PA:19,AB:18,H:2,HR:0,R:1,RBI:3,BB:1,SO:8,SB:0,AVG:0.111,OBP:0.158}},
  {id:197,name:"Jordan Geber",team:"Israel",pool:"Pool D",pos:"P",mlbId:802363,bats:"R",throws:"R",num:33,dob:"1999-07-31",age:27,h:191,w:93,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:1,H:0,SV:0,IP:3.2,SO:2,BB:1,HA:5,HR:1,ER:4,ERA:9.82,WHIP:1.64,Kpct:11.1,BBpct:5.6}},
  {id:198,name:"Josh Mallitz",team:"Israel",pool:"Pool D",pos:"P",mlbId:695256,bats:"R",throws:"R",num:58,dob:"2001-10-20",age:25,h:191,w:95,isPitcher:true,wbcStats:{G:3,GS:0,W:1,L:0,H:1,SV:0,IP:2.2,SO:1,BB:1,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.75,Kpct:10.0,BBpct:10.0}},
  {id:199,name:"Matt Bowman",team:"Israel",pool:"Pool D",pos:"P",mlbId:621199,bats:"R",throws:"R",num:66,dob:"1991-05-31",age:35,h:183,w:84,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:4.0,SO:1,BB:1,HA:2,HR:0,ER:0,ERA:0.0,WHIP:0.75,Kpct:7.7,BBpct:7.7}},
  {id:200,name:"Matt Mervis",team:"Israel",pool:"Pool D",pos:"1B",mlbId:670223,bats:"L",throws:"R",num:52,dob:"1998-04-16",age:28,h:188,w:102,isPitcher:false,wbcStats:{PA:12,AB:12,H:3,HR:0,R:0,RBI:3,BB:0,SO:5,SB:0,AVG:0.25,OBP:0.25}},
  {id:201,name:"Max Lazar",team:"Israel",pool:"Pool D",pos:"P",mlbId:676661,bats:"R",throws:"R",num:60,dob:"1999-06-03",age:27,h:185,w:91,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.2,SO:2,BB:0,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.0,Kpct:40.0,BBpct:0.0}},
  {id:202,name:"Noah Mendlinger",team:"Israel",pool:"Pool D",pos:"3B",mlbId:702331,bats:"L",throws:"R",num:3,dob:"2000-08-09",age:26,h:170,w:82,isPitcher:false,wbcStats:{PA:22,AB:18,H:5,HR:0,R:3,RBI:1,BB:4,SO:2,SB:0,AVG:0.278,OBP:0.409}},
  {id:203,name:"RJ Schreck",team:"Israel",pool:"Pool D",pos:"LF",mlbId:702302,bats:"L",throws:"R",num:0,dob:"2000-07-12",age:26,h:183,w:93,isPitcher:false,wbcStats:{PA:23,AB:19,H:3,HR:1,R:3,RBI:3,BB:4,SO:6,SB:0,AVG:0.158,OBP:0.304}},
  {id:204,name:"Rob Kaminsky",team:"Israel",pool:"Pool D",pos:"P",mlbId:641739,bats:"R",throws:"L",num:75,dob:"1994-09-02",age:32,h:183,w:88,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:2.2,SO:1,BB:2,HA:4,HR:1,ER:1,ERA:3.38,WHIP:2.25,Kpct:8.3,BBpct:16.7}},
  {id:205,name:"Ryan Prager",team:"Israel",pool:"Pool D",pos:"P",mlbId:696492,bats:"L",throws:"L",num:12,dob:"2002-10-26",age:24,h:191,w:91,isPitcher:true,wbcStats:{G:2,GS:1,W:0,L:1,H:0,SV:0,IP:3.1,SO:4,BB:6,HA:2,HR:2,ER:6,ERA:16.2,WHIP:2.4,Kpct:22.2,BBpct:33.3}},
  {id:206,name:"Spencer Horwitz",team:"Israel",pool:"Pool D",pos:"1B",mlbId:687462,bats:"L",throws:"R",num:13,dob:"1997-11-14",age:29,h:178,w:93,isPitcher:false,wbcStats:{PA:21,AB:19,H:4,HR:1,R:3,RBI:1,BB:0,SO:4,SB:0,AVG:0.211,OBP:0.211}},
  {id:207,name:"Tanner Jacobson",team:"Israel",pool:"Pool D",pos:"P",mlbId:805374,bats:"R",throws:"R",num:39,dob:"2000-01-24",age:26,h:185,w:86,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:1.1,SO:0,BB:5,HA:3,HR:0,ER:3,ERA:20.25,WHIP:6.0,Kpct:0.0,BBpct:45.5}},
  {id:208,name:"Tommy Kahnle",team:"Israel",pool:"Pool D",pos:"P",mlbId:592454,bats:"R",throws:"R",num:43,dob:"1989-08-07",age:37,h:185,w:104,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:3.0,SO:3,BB:1,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.67,Kpct:27.3,BBpct:9.1}},
  {id:209,name:"Zach Levenson",team:"Israel",pool:"Pool D",pos:"OF",mlbId:804241,bats:"R",throws:"R",num:10,dob:"2002-03-06",age:24,h:183,w:96,isPitcher:false,wbcStats:{PA:23,AB:21,H:5,HR:1,R:1,RBI:3,BB:2,SO:5,SB:1,AVG:0.238,OBP:0.304}},
  {id:210,name:"Zack Leban",team:"Israel",pool:"Pool D",pos:"P",mlbId:681671,bats:"R",throws:"R",num:53,dob:"1996-05-30",age:30,h:188,w:107,isPitcher:true,wbcStats:{G:4,GS:0,W:0,L:0,H:0,SV:0,IP:2.0,SO:1,BB:0,HA:3,HR:0,ER:1,ERA:4.5,WHIP:1.5,Kpct:11.1,BBpct:0.0}},
  {id:211,name:"Zack Weiss",team:"Israel",pool:"Pool D",pos:"P",mlbId:592848,bats:"R",throws:"R",num:48,dob:"1992-06-16",age:34,h:191,w:95,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:2.2,SO:3,BB:3,HA:2,HR:1,ER:2,ERA:6.75,WHIP:1.88,Kpct:27.3,BBpct:27.3}},
  {id:212,name:"Andre Scrubb",team:"Great Britain",pool:"Pool B",pos:"P",mlbId:668687,bats:"R",throws:"R",num:70,dob:"1995-01-13",age:31,h:193,w:122,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:1,H:0,SV:0,IP:0.0,SO:0,BB:0,HA:3,HR:1,ER:3,Kpct:0.0,BBpct:0.0}},
  {id:213,name:"BJ Murray",team:"Great Britain",pool:"Pool B",pos:"3B",mlbId:681230,bats:"S",throws:"R",num:7,dob:"2000-01-05",age:26,h:180,w:93,isPitcher:false,wbcStats:{PA:23,AB:19,H:7,HR:0,R:2,RBI:0,BB:4,SO:5,SB:2,AVG:0.368,OBP:0.478}},
  {id:214,name:"Brendan Beck",team:"Great Britain",pool:"Pool B",pos:"P",mlbId:694341,bats:"R",throws:"R",num:19,dob:"1998-10-06",age:28,h:188,w:99,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:4.0,SO:4,BB:2,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.5,Kpct:28.6,BBpct:14.3}},
  {id:215,name:"Chavez Fernander",team:"Great Britain",pool:"Pool B",pos:"P",mlbId:681501,bats:"R",throws:"R",num:46,dob:"1997-07-07",age:29,h:191,w:93,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:1.1,SO:0,BB:3,HA:3,HR:0,ER:4,ERA:27.0,WHIP:4.5,Kpct:0.0,BBpct:30.0}},
  {id:216,name:"Donovan Benoit",team:"Great Britain",pool:"Pool B",pos:"P",mlbId:687630,bats:"R",throws:"R",num:41,dob:"1999-01-22",age:27,h:191,w:90,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:2.0,SO:0,BB:2,HA:2,HR:0,ER:1,ERA:4.5,WHIP:2.0,Kpct:0.0,BBpct:16.7}},
  {id:217,name:"Dylan Covey",team:"Great Britain",pool:"Pool B",pos:"P",mlbId:592229,bats:"R",throws:"R",num:33,dob:"1991-08-14",age:35,h:185,w:98,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:2.0,SO:1,BB:1,HA:3,HR:0,ER:2,ERA:9.0,WHIP:2.0,Kpct:10.0,BBpct:10.0}},
  {id:218,name:"Gary Gill Hill",team:"Great Britain",pool:"Pool B",pos:"P",mlbId:804541,bats:"R",throws:"R",num:17,dob:"2004-09-20",age:22,h:188,w:73,isPitcher:true,wbcStats:{G:2,GS:0,W:1,L:1,H:0,SV:0,IP:3.2,SO:5,BB:2,HA:1,HR:0,ER:3,ERA:7.36,WHIP:0.82,Kpct:33.3,BBpct:13.3}},
  {id:219,name:"Graham Spraker",team:"Great Britain",pool:"Pool B",pos:"P",mlbId:676942,bats:"R",throws:"R",num:96,dob:"1995-03-19",age:31,h:188,w:85,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:4.0,SO:4,BB:3,HA:3,HR:0,ER:3,ERA:6.75,WHIP:1.5,Kpct:25.0,BBpct:18.8}},
  {id:220,name:"Harry Ford",team:"Great Britain",pool:"Pool B",pos:"C",mlbId:695670,bats:"R",throws:"R",num:1,dob:"2003-02-21",age:23,h:178,w:91,isPitcher:false,wbcStats:{PA:25,AB:23,H:7,HR:2,R:2,RBI:5,BB:2,SO:7,SB:0,AVG:0.304,OBP:0.36}},
  {id:221,name:"Ian Lewis Jr.",team:"Great Britain",pool:"Pool B",pos:"2B",mlbId:691588,bats:"S",throws:"R",num:6,dob:"2003-02-04",age:23,h:180,w:80,isPitcher:false,wbcStats:{PA:21,AB:21,H:3,HR:1,R:2,RBI:2,BB:0,SO:3,SB:0,AVG:0.143,OBP:0.143}},
  {id:222,name:"Ivan Johnson",team:"Great Britain",pool:"Pool B",pos:"2B",mlbId:671155,bats:"S",throws:"R",num:15,dob:"1998-10-11",age:28,h:183,w:86,isPitcher:false,wbcStats:{PA:20,AB:16,H:2,HR:0,R:4,RBI:0,BB:4,SO:5,SB:1,AVG:0.125,OBP:0.3}},
  {id:223,name:"Jack Anderson",team:"Great Britain",pool:"Pool B",pos:"P",mlbId:681252,bats:"R",throws:"R",num:62,dob:"1999-11-23",age:27,h:191,w:89,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:3.0,SO:3,BB:0,HA:2,HR:1,ER:1,ERA:3.0,WHIP:0.67,Kpct:27.3,BBpct:0.0}},
  {id:224,name:"Jack Seppings",team:"Great Britain",pool:"Pool B",pos:"P",mlbId:808952,bats:"R",throws:"R",num:27,dob:"2002-07-03",age:24,h:185,w:86,isPitcher:true,wbcStats:{G:2,GS:0,W:1,L:0,H:0,SV:0,IP:3.0,SO:1,BB:1,HA:5,HR:1,ER:3,ERA:9.0,WHIP:2.0,Kpct:6.7,BBpct:6.7}},
  {id:225,name:"Jazz Chisholm Jr.",team:"Great Britain",pool:"Pool B",pos:"2B",mlbId:665862,bats:"L",throws:"R",num:3,dob:"1998-02-01",age:28,h:180,w:83,isPitcher:false,wbcStats:{PA:26,AB:23,H:5,HR:1,R:4,RBI:5,BB:3,SO:8,SB:1,AVG:0.217,OBP:0.308}},
  {id:226,name:"Justin Wylie",team:"Great Britain",pool:"Pool B",pos:"OF",mlbId:687045,bats:"R",throws:"R",num:13,dob:"1996-08-26",age:30,h:178,w:88,isPitcher:false,wbcStats:{PA:5,AB:5,H:1,HR:0,R:0,RBI:0,BB:0,SO:2,SB:0,AVG:0.2,OBP:0.2}},
  {id:227,name:"Kristian Robinson",team:"Great Britain",pool:"Pool B",pos:"OF",mlbId:677565,bats:"R",throws:"R",num:59,dob:"2000-12-11",age:26,h:191,w:86,isPitcher:false,wbcStats:{PA:14,AB:13,H:3,HR:0,R:1,RBI:2,BB:1,SO:6,SB:1,AVG:0.231,OBP:0.286}},
  {id:228,name:"Matt Koperniak",team:"Great Britain",pool:"Pool B",pos:"OF",mlbId:689288,bats:"L",throws:"R",num:29,dob:"1998-02-08",age:28,h:183,w:94,isPitcher:false,wbcStats:{PA:21,AB:17,H:6,HR:0,R:0,RBI:5,BB:4,SO:4,SB:0,AVG:0.353,OBP:0.476}},
  {id:229,name:"Miles Langhorne",team:"Great Britain",pool:"Pool B",pos:"P",mlbId:695680,bats:"R",throws:"R",num:45,dob:"2003-04-30",age:23,h:193,w:98,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.2,SO:2,BB:1,HA:2,HR:0,ER:0,ERA:0.0,WHIP:1.8,Kpct:25.0,BBpct:12.5}},
  {id:230,name:"Najer Victor",team:"Great Britain",pool:"Pool B",pos:"P",mlbId:695402,bats:"R",throws:"R",num:8,dob:"2001-11-28",age:25,h:185,w:88,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:3.1,SO:7,BB:2,HA:2,HR:1,ER:1,ERA:2.7,WHIP:1.2,Kpct:46.7,BBpct:13.3}},
  {id:231,name:"Nate Eaton",team:"Great Britain",pool:"Pool B",pos:"3B",mlbId:681987,bats:"R",throws:"R",num:18,dob:"1996-12-22",age:30,h:180,w:91,isPitcher:false,wbcStats:{PA:27,AB:26,H:7,HR:1,R:5,RBI:1,BB:1,SO:6,SB:2,AVG:0.269,OBP:0.296}},
  {id:232,name:"Nick Ward",team:"Great Britain",pool:"Pool B",pos:"SS",mlbId:682187,bats:"L",throws:"R",num:5,dob:"1995-10-19",age:31,h:175,w:82,isPitcher:false,wbcStats:{PA:8,AB:8,H:1,HR:0,R:1,RBI:0,BB:0,SO:2,SB:0,AVG:0.125,OBP:0.125}},
  {id:233,name:"Nick Wells",team:"Great Britain",pool:"Pool B",pos:"P",mlbId:657101,bats:"L",throws:"L",num:23,dob:"1996-02-21",age:30,h:196,w:84,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:1,H:0,SV:0,IP:3.2,SO:3,BB:1,HA:4,HR:0,ER:1,ERA:2.45,WHIP:1.36,Kpct:17.6,BBpct:5.9}},
  {id:234,name:"Owen Wild",team:"Great Britain",pool:"Pool B",pos:"P",mlbId:801643,bats:"R",throws:"R",num:34,dob:"2002-07-30",age:24,h:188,w:104,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:3.0,SO:2,BB:1,HA:2,HR:2,ER:2,ERA:6.0,WHIP:1.0,Kpct:16.7,BBpct:8.3}},
  {id:235,name:"Ryan Long",team:"Great Britain",pool:"Pool B",pos:"P",mlbId:702465,bats:"R",throws:"R",num:35,dob:"1999-10-19",age:27,h:198,w:109,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:3.0,SO:4,BB:1,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.33,Kpct:36.4,BBpct:9.1}},
  {id:236,name:"Trayce Thompson",team:"Great Britain",pool:"Pool B",pos:"OF",mlbId:572204,bats:"R",throws:"R",num:28,dob:"1991-03-15",age:35,h:188,w:102,isPitcher:false,wbcStats:{PA:23,AB:19,H:3,HR:0,R:2,RBI:1,BB:4,SO:9,SB:1,AVG:0.158,OBP:0.304}},
  {id:237,name:"Tristan Beck",team:"Great Britain",pool:"Pool B",pos:"P",mlbId:663941,bats:"R",throws:"R",num:43,dob:"1996-06-24",age:30,h:193,w:92,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.1,SO:1,BB:0,HA:1,HR:1,ER:1,ERA:6.75,WHIP:0.75,Kpct:20.0,BBpct:0.0}},
  {id:238,name:"Tyler Viza",team:"Great Britain",pool:"Pool B",pos:"P",mlbId:643588,bats:"R",throws:"R",num:21,dob:"1994-10-21",age:32,h:191,w:77,isPitcher:true,wbcStats:{G:2,GS:1,W:0,L:0,H:0,SV:0,IP:4.0,SO:0,BB:0,HA:4,HR:0,ER:0,ERA:0.0,WHIP:1.0,Kpct:0.0,BBpct:0.0}},
  {id:239,name:"Vance Worley",team:"Great Britain",pool:"Pool B",pos:"P",mlbId:474699,bats:"R",throws:"R",num:49,dob:"1987-09-25",age:39,h:188,w:109,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:3.0,SO:1,BB:5,HA:3,HR:0,ER:1,ERA:3.0,WHIP:2.67,Kpct:6.2,BBpct:31.2}},
  {id:240,name:"Wallace Clark",team:"Great Britain",pool:"Pool B",pos:"SS",mlbId:800712,bats:"S",throws:"R",num:10,dob:"2002-06-08",age:24,h:183,w:88,isPitcher:false,wbcStats:{PA:5,AB:3,H:0,HR:0,R:1,RBI:0,BB:2,SO:2,SB:0,AVG:0.0,OBP:0.4}},
  {id:241,name:"Will Cresswell",team:"Great Britain",pool:"Pool B",pos:"C",mlbId:701796,bats:"R",throws:"R",num:30,dob:"2003-08-18",age:23,h:183,w:93,isPitcher:false,wbcStats:{PA:5,AB:5,H:0,HR:0,R:0,RBI:0,BB:0,SO:3,SB:0,AVG:0.0,OBP:0.0}},
  {id:242,name:"Antwone Kelly",team:"Netherlands",pool:"Pool D",pos:"P",mlbId:699008,bats:"R",throws:"R",num:58,dob:"2003-09-01",age:23,h:178,w:108,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:1,H:0,SV:0,IP:3.0,SO:1,BB:1,HA:4,HR:1,ER:2,ERA:6.0,WHIP:1.67,Kpct:7.7,BBpct:7.7}},
  {id:243,name:"Arij Fransen",team:"Netherlands",pool:"Pool D",pos:"P",mlbId:687677,bats:"R",throws:"R",num:17,dob:"2001-05-20",age:25,h:191,w:98,isPitcher:true,wbcStats:{G:2,GS:1,W:0,L:1,H:0,SV:0,IP:3.0,SO:3,BB:4,HA:3,HR:0,ER:4,ERA:12.0,WHIP:2.33,Kpct:18.8,BBpct:25.0}},
  {id:244,name:"Ceddanne Rafaela",team:"Netherlands",pool:"Pool D",pos:"OF",mlbId:678882,bats:"R",throws:"R",num:3,dob:"2000-09-18",age:26,h:178,w:75,isPitcher:false,wbcStats:{PA:26,AB:25,H:8,HR:2,R:5,RBI:6,BB:0,SO:1,SB:1,AVG:0.32,OBP:0.32}},
  {id:245,name:"Chadwick Tromp",team:"Netherlands",pool:"Pool D",pos:"C",mlbId:644433,bats:"R",throws:"R",num:14,dob:"1995-03-21",age:31,h:173,w:100,isPitcher:false,wbcStats:{PA:25,AB:20,H:2,HR:0,R:0,RBI:2,BB:4,SO:6,SB:0,AVG:0.1,OBP:0.25}},
  {id:246,name:"Dayson Croes",team:"Netherlands",pool:"Pool D",pos:"OF",mlbId:806401,bats:"L",throws:"R",num:5,dob:"1999-10-08",age:27,h:178,w:93,isPitcher:false,wbcStats:{PA:9,AB:9,H:1,HR:0,R:0,RBI:2,BB:0,SO:4,SB:0,AVG:0.111,OBP:0.111}},
  {id:247,name:"Delano Selassa",team:"Netherlands",pool:"Pool D",pos:"OF",mlbId:838330,bats:"R",throws:"R",num:22,dob:"1999-10-25",age:27,h:188,w:86,isPitcher:false,wbcStats:{PA:3,AB:3,H:0,HR:0,R:0,RBI:0,BB:0,SO:2,SB:0,AVG:0.0,OBP:0.0}},
  {id:248,name:"Derek West",team:"Netherlands",pool:"Pool D",pos:"P",mlbId:663723,bats:"R",throws:"R",num:0,dob:"1996-12-02",age:30,h:196,w:117,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:5.2,SO:2,BB:3,HA:5,HR:0,ER:0,ERA:0.0,WHIP:1.41,Kpct:8.3,BBpct:12.5}},
  {id:249,name:"Didi Gregorius",team:"Netherlands",pool:"Pool D",pos:"SS",mlbId:544369,bats:"L",throws:"R",num:18,dob:"1990-02-18",age:36,h:191,w:93,isPitcher:false,wbcStats:{PA:22,AB:20,H:6,HR:1,R:4,RBI:4,BB:0,SO:2,SB:0,AVG:0.3,OBP:0.3}},
  {id:250,name:"Druw Jones",team:"Netherlands",pool:"Pool D",pos:"OF",mlbId:702258,bats:"R",throws:"R",num:4,dob:"2003-11-28",age:23,h:188,w:82,isPitcher:false,wbcStats:{PA:26,AB:19,H:4,HR:0,R:3,RBI:2,BB:6,SO:10,SB:1,AVG:0.211,OBP:0.4}},
  {id:251,name:"Dylan Wilson",team:"Netherlands",pool:"Pool D",pos:"P",mlbId:808511,bats:"R",throws:"R",num:20,dob:"2005-12-01",age:21,h:183,w:73,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:2.0,SO:1,BB:1,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.5,Kpct:14.3,BBpct:14.3}},
  {id:252,name:"Eric Mendez",team:"Netherlands",pool:"Pool D",pos:"P",mlbId:677804,bats:"R",throws:"R",num:29,dob:"1999-12-03",age:27,h:183,w:79,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:3.0,SO:2,BB:3,HA:5,HR:1,ER:2,ERA:6.0,WHIP:2.67,Kpct:13.3,BBpct:20.0}},
  {id:253,name:"Hendrik Clementina",team:"Netherlands",pool:"Pool D",pos:"C",mlbId:649955,bats:"R",throws:"R",num:12,dob:"1997-06-17",age:29,h:188,w:113,isPitcher:false,wbcStats:{PA:25,AB:17,H:5,HR:0,R:3,RBI:1,BB:7,SO:4,SB:1,AVG:0.294,OBP:0.5}},
  {id:254,name:"Jaitoine Kelly",team:"Netherlands",pool:"Pool D",pos:"P",mlbId:823858,bats:"R",throws:"R",num:34,dob:"2007-06-29",age:19,h:191,w:117,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:2.0,SO:3,BB:1,HA:1,HR:0,ER:0,ERA:0.0,WHIP:1.0,Kpct:33.3,BBpct:11.1}},
  {id:255,name:"Jakey Josepha",team:"Netherlands",pool:"Pool D",pos:"CF",mlbId:699228,bats:"L",throws:"R",num:37,dob:"2004-05-15",age:22,h:188,w:61,isPitcher:false,wbcStats:{PA:4,AB:4,H:1,HR:0,R:0,RBI:0,BB:0,SO:3,SB:0,AVG:0.25,OBP:0.25}},
  {id:256,name:"Jamdrick Cornelia",team:"Netherlands",pool:"Pool D",pos:"P",mlbId:808378,bats:"L",throws:"L",num:30,dob:"2005-11-17",age:21,h:183,w:64,isPitcher:true,wbcStats:{G:1,GS:1,W:1,L:0,H:0,SV:0,IP:1.0,SO:0,BB:1,HA:0,HR:0,ER:2,ERA:18.0,WHIP:1.0,Kpct:0.0,BBpct:20.0}},
  {id:257,name:"Jaydenn Estanista",team:"Netherlands",pool:"Pool D",pos:"P",mlbId:692838,bats:"R",throws:"R",num:32,dob:"2001-10-03",age:25,h:191,w:82,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:0.1,SO:0,BB:3,HA:4,HR:0,ER:6,ERA:162.0,WHIP:21.0,Kpct:0.0,BBpct:33.3}},
  {id:258,name:"Juan Carlos Sulbaran",team:"Netherlands",pool:"Pool D",pos:"P",mlbId:543833,bats:"R",throws:"R",num:45,dob:"1989-11-09",age:37,h:188,w:93,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:2.2,SO:1,BB:1,HA:3,HR:1,ER:2,ERA:6.75,WHIP:1.5,Kpct:9.1,BBpct:9.1}},
  {id:259,name:"Juremi Profar",team:"Netherlands",pool:"Pool D",pos:"3B",mlbId:642282,bats:"R",throws:"R",num:13,dob:"1996-01-30",age:30,h:185,w:84,isPitcher:false,wbcStats:{PA:8,AB:7,H:3,HR:0,R:0,RBI:0,BB:1,SO:2,SB:0,AVG:0.429,OBP:0.5}},
  {id:260,name:"Justin Morales",team:"Netherlands",pool:"Pool D",pos:"P",mlbId:838329,bats:"R",throws:"R",num:46,dob:"2004-12-14",age:22,h:193,w:95,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.2,SO:1,BB:4,HA:1,HR:0,ER:0,ERA:0.0,WHIP:3.0,Kpct:10.0,BBpct:40.0}},
  {id:261,name:"Kenley Jansen",team:"Netherlands",pool:"Pool D",pos:"P",mlbId:445276,bats:"S",throws:"R",num:74,dob:"1987-09-30",age:39,h:196,w:120,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:2.0,SO:1,BB:1,HA:3,HR:0,ER:1,ERA:4.5,WHIP:2.0,Kpct:11.1,BBpct:11.1}},
  {id:262,name:"Kevin Kelly",team:"Netherlands",pool:"Pool D",pos:"P",mlbId:808721,bats:"R",throws:"R",num:33,dob:"1990-05-27",age:36,h:183,w:91,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:1,H:0,SV:0,IP:5.0,SO:3,BB:2,HA:7,HR:0,ER:2,ERA:3.6,WHIP:1.8,Kpct:12.0,BBpct:8.0}},
  {id:263,name:"Lars Huijer",team:"Netherlands",pool:"Pool D",pos:"P",mlbId:609044,bats:"R",throws:"R",num:16,dob:"1993-09-22",age:33,h:193,w:93,isPitcher:true,wbcStats:{G:3,GS:0,W:1,L:0,H:1,SV:0,IP:5.0,SO:5,BB:0,HA:3,HR:1,ER:2,ERA:3.6,WHIP:0.6,Kpct:26.3,BBpct:0.0}},
  {id:264,name:"Ozzie Albies",team:"Netherlands",pool:"Pool D",pos:"2B",mlbId:645277,bats:"S",throws:"R",num:1,dob:"1997-01-07",age:29,h:170,w:75,isPitcher:false,wbcStats:{PA:24,AB:23,H:6,HR:2,R:4,RBI:6,BB:0,SO:3,SB:0,AVG:0.261,OBP:0.261}},
  {id:265,name:"Ray Patrick Didder",team:"Netherlands",pool:"Pool D",pos:"OF",mlbId:642720,bats:"R",throws:"R",num:11,dob:"1994-10-01",age:32,h:178,w:90,isPitcher:false,wbcStats:{PA:22,AB:19,H:5,HR:1,R:4,RBI:1,BB:2,SO:5,SB:0,AVG:0.263,OBP:0.333}},
  {id:266,name:"Ryjeteri Merite",team:"Netherlands",pool:"Pool D",pos:"P",mlbId:820853,bats:"L",throws:"L",num:52,dob:"2005-12-16",age:21,h:191,w:68,isPitcher:true,wbcStats:{G:3,GS:2,W:1,L:0,H:0,SV:0,IP:3.2,SO:3,BB:2,HA:3,HR:1,ER:3,ERA:7.36,WHIP:1.36,Kpct:18.8,BBpct:12.5}},
  {id:267,name:"Shairon Martis",team:"Netherlands",pool:"Pool D",pos:"P",mlbId:463017,bats:"R",throws:"R",num:39,dob:"1987-03-30",age:39,h:185,w:102,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:4.1,SO:2,BB:1,HA:4,HR:0,ER:2,ERA:4.15,WHIP:1.15,Kpct:11.1,BBpct:5.6}},
  {id:268,name:"Sharlon Schoop",team:"Netherlands",pool:"Pool D",pos:"SS",mlbId:463748,bats:"R",throws:"R",num:15,dob:"1987-04-15",age:39,h:188,w:86,isPitcher:false,wbcStats:{PA:12,AB:9,H:1,HR:0,R:1,RBI:1,BB:2,SO:4,SB:0,AVG:0.111,OBP:0.273}},
  {id:269,name:"Shawndrick Oduber",team:"Netherlands",pool:"Pool D",pos:"P",mlbId:806779,bats:"R",throws:"R",num:35,dob:"2004-12-16",age:22,h:183,w:77,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:0.1,SO:0,BB:3,HA:0,HR:0,ER:2,ERA:54.0,WHIP:9.0,Kpct:0.0,BBpct:75.0}},
  {id:270,name:"Wendell Floranus",team:"Netherlands",pool:"Pool D",pos:"P",mlbId:622464,bats:"R",throws:"R",num:99,dob:"1995-04-16",age:31,h:183,w:73,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:3.1,SO:4,BB:3,HA:6,HR:3,ER:6,ERA:16.2,WHIP:2.7,Kpct:21.1,BBpct:15.8}},
  {id:271,name:"Xander Bogaerts",team:"Netherlands",pool:"Pool D",pos:"SS",mlbId:593428,bats:"R",throws:"R",num:2,dob:"1992-10-01",age:34,h:188,w:99,isPitcher:false,wbcStats:{PA:25,AB:23,H:7,HR:0,R:3,RBI:2,BB:1,SO:5,SB:1,AVG:0.304,OBP:0.333}},
  {id:272,name:"Andres Gimenez",team:"Venezuela",pool:"Pool D",pos:"2B",mlbId:665926,bats:"L",throws:"R",num:0,dob:"1998-09-04",age:28,h:180,w:73,isPitcher:false,wbcStats:{PA:10,AB:7,H:1,HR:0,R:1,RBI:0,BB:2,SO:1,SB:0,AVG:0.143,OBP:0.333}},
  {id:273,name:"Andres Machado",team:"Venezuela",pool:"Pool D",pos:"P",mlbId:600921,bats:"R",throws:"R",num:30,dob:"1993-04-22",age:33,h:185,w:105,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:3.0,SO:7,BB:1,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.67,Kpct:58.3,BBpct:8.3}},
  {id:274,name:"Angel Zerpa",team:"Venezuela",pool:"Pool D",pos:"P",mlbId:672582,bats:"L",throws:"L",num:61,dob:"1999-09-27",age:27,h:183,w:108,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:3.0,SO:2,BB:0,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.0,Kpct:22.2,BBpct:0.0}},
  {id:275,name:"Anthony Molina",team:"Venezuela",pool:"Pool D",pos:"P",mlbId:683627,bats:"R",throws:"R",num:35,dob:"2002-01-12",age:24,h:185,w:77,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:2.0,SO:4,BB:0,HA:3,HR:1,ER:1,ERA:4.5,WHIP:1.5,Kpct:44.4,BBpct:0.0}},
  {id:276,name:"Antonio Senzatela",team:"Venezuela",pool:"Pool D",pos:"P",mlbId:622608,bats:"R",throws:"R",num:34,dob:"1995-01-21",age:31,h:185,w:107,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:3.0,SO:4,BB:0,HA:2,HR:0,ER:0,ERA:0.0,WHIP:0.67,Kpct:36.4,BBpct:0.0}},
  {id:277,name:"Carlos Guzman",team:"Venezuela",pool:"Pool D",pos:"P",mlbId:664346,bats:"R",throws:"R",num:20,dob:"1998-05-16",age:28,h:185,w:95,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.2,SO:1,BB:1,HA:1,HR:0,ER:0,ERA:0.0,WHIP:1.2,Kpct:14.3,BBpct:14.3}},
  {id:278,name:"Christian Suarez",team:"Venezuela",pool:"Pool D",pos:"P",mlbId:682949,bats:"L",throws:"L",num:73,dob:"2000-11-25",age:26,h:180,w:73,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:1.1,SO:0,BB:1,HA:3,HR:1,ER:3,ERA:20.25,WHIP:3.0,Kpct:0.0,BBpct:12.5}},
  {id:279,name:"Daniel Palencia",team:"Venezuela",pool:"Pool D",pos:"P",mlbId:694037,bats:"R",throws:"R",num:29,dob:"2000-02-05",age:26,h:180,w:73,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.2,SO:3,BB:1,HA:3,HR:0,ER:1,ERA:5.4,WHIP:2.4,Kpct:33.3,BBpct:11.1}},
  {id:280,name:"Eduard Bazardo",team:"Venezuela",pool:"Pool D",pos:"P",mlbId:660825,bats:"R",throws:"R",num:83,dob:"1995-09-01",age:31,h:183,w:86,isPitcher:true,wbcStats:{G:2,GS:1,W:0,L:0,H:1,SV:0,IP:2.0,SO:0,BB:0,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.5,Kpct:0.0,BBpct:0.0}},
  {id:281,name:"Eduardo Rodriguez",team:"Venezuela",pool:"Pool D",pos:"P",mlbId:593958,bats:"L",throws:"L",num:52,dob:"1993-04-07",age:33,h:188,w:105,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:1,H:0,SV:0,IP:1.2,SO:2,BB:3,HA:1,HR:0,ER:2,ERA:10.8,WHIP:2.4,Kpct:22.2,BBpct:33.3}},
  {id:282,name:"Enmanuel De Jesus",team:"Venezuela",pool:"Pool D",pos:"P",mlbId:646241,bats:"L",throws:"L",num:37,dob:"1996-12-10",age:30,h:191,w:86,isPitcher:true,wbcStats:{G:1,GS:1,W:1,L:0,H:0,SV:0,IP:5.0,SO:8,BB:0,HA:2,HR:0,ER:1,ERA:1.8,WHIP:0.4,Kpct:47.1,BBpct:0.0}},
  {id:283,name:"Eugenio Suarez",team:"Venezuela",pool:"Pool D",pos:"3B",mlbId:553993,bats:"R",throws:"R",num:7,dob:"1991-07-18",age:35,h:180,w:97,isPitcher:false,wbcStats:{PA:16,AB:15,H:2,HR:1,R:1,RBI:2,BB:1,SO:1,SB:0,AVG:0.133,OBP:0.188}},
  {id:284,name:"Ezequiel Tovar",team:"Venezuela",pool:"Pool D",pos:"SS",mlbId:678662,bats:"R",throws:"R",num:14,dob:"2001-08-01",age:25,h:183,w:73,isPitcher:false,wbcStats:{PA:8,AB:7,H:4,HR:0,R:1,RBI:0,BB:1,SO:2,SB:1,AVG:0.571,OBP:0.625}},
  {id:285,name:"Gleyber Torres",team:"Venezuela",pool:"Pool D",pos:"2B",mlbId:650402,bats:"R",throws:"R",num:25,dob:"1996-12-13",age:30,h:178,w:93,isPitcher:false,wbcStats:{PA:12,AB:10,H:1,HR:0,R:1,RBI:0,BB:2,SO:3,SB:1,AVG:0.1,OBP:0.25}},
  {id:286,name:"Jackson Chourio",team:"Venezuela",pool:"Pool D",pos:"CF",mlbId:694192,bats:"R",throws:"R",num:1,dob:"2004-03-11",age:22,h:185,w:90,isPitcher:false,wbcStats:{PA:10,AB:7,H:1,HR:0,R:0,RBI:1,BB:1,SO:2,SB:0,AVG:0.143,OBP:0.25}},
  {id:287,name:"Javier Sanoja",team:"Venezuela",pool:"Pool D",pos:"OF",mlbId:691594,bats:"R",throws:"R",num:4,dob:"2002-09-03",age:24,h:170,w:68,isPitcher:false,wbcStats:{PA:13,AB:13,H:3,HR:1,R:1,RBI:1,BB:0,SO:2,SB:0,AVG:0.231,OBP:0.231}},
  {id:288,name:"Jhonathan Diaz",team:"Venezuela",pool:"Pool D",pos:"P",mlbId:646242,bats:"L",throws:"L",num:74,dob:"1996-09-13",age:30,h:183,w:77,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:1,SV:0,IP:1.0,SO:1,BB:1,HA:1,HR:0,ER:0,ERA:0.0,WHIP:2.0,Kpct:25.0,BBpct:25.0}},
  {id:289,name:"Jose Butto",team:"Venezuela",pool:"Pool D",pos:"P",mlbId:676130,bats:"R",throws:"R",num:70,dob:"1998-03-19",age:28,h:185,w:92,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:4.0,SO:2,BB:0,HA:2,HR:0,ER:0,ERA:0.0,WHIP:0.5,Kpct:13.3,BBpct:0.0}},
  {id:290,name:"Keider Montero",team:"Venezuela",pool:"Pool D",pos:"P",mlbId:672456,bats:"R",throws:"R",num:54,dob:"2000-07-06",age:26,h:185,w:66,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:1,IP:6.0,SO:5,BB:0,HA:5,HR:0,ER:0,ERA:0.0,WHIP:0.83,Kpct:22.7,BBpct:0.0}},
  {id:291,name:"Luinder Avila",team:"Venezuela",pool:"Pool D",pos:"P",mlbId:679883,bats:"R",throws:"R",num:58,dob:"2001-08-21",age:25,h:191,w:88,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:1,SV:0,IP:2.0,SO:2,BB:3,HA:1,HR:0,ER:0,ERA:0.0,WHIP:2.0,Kpct:18.2,BBpct:27.3}},
  {id:292,name:"Luis Arraez",team:"Venezuela",pool:"Pool D",pos:"1B",mlbId:650333,bats:"L",throws:"R",num:2,dob:"1997-04-09",age:29,h:178,w:79,isPitcher:false,wbcStats:{PA:19,AB:18,H:9,HR:2,R:6,RBI:7,BB:1,SO:1,SB:0,AVG:0.5,OBP:0.526}},
  {id:293,name:"Maikel Garcia",team:"Venezuela",pool:"Pool D",pos:"3B",mlbId:672580,bats:"R",throws:"R",num:11,dob:"2000-03-03",age:26,h:185,w:82,isPitcher:false,wbcStats:{PA:15,AB:14,H:3,HR:0,R:2,RBI:3,BB:1,SO:5,SB:1,AVG:0.214,OBP:0.267}},
  {id:294,name:"Ranger Suarez",team:"Venezuela",pool:"Pool D",pos:"P",mlbId:624133,bats:"L",throws:"L",num:55,dob:"1995-08-26",age:31,h:185,w:98,isPitcher:true,wbcStats:{G:1,GS:1,W:1,L:0,H:0,SV:0,IP:2.0,SO:1,BB:1,HA:3,HR:0,ER:1,ERA:4.5,WHIP:2.0,Kpct:10.0,BBpct:10.0}},
  {id:295,name:"Ricardo Sanchez",team:"Venezuela",pool:"Pool D",pos:"P",mlbId:645307,bats:"L",throws:"L",num:64,dob:"1997-04-11",age:29,h:178,w:100,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:3.0,SO:1,BB:1,HA:2,HR:1,ER:1,ERA:3.0,WHIP:1.0,Kpct:8.3,BBpct:8.3}},
  {id:296,name:"Ronald Acuna Jr.",team:"Venezuela",pool:"Pool D",pos:"OF",mlbId:660670,bats:"R",throws:"R",num:21,dob:"1997-12-18",age:29,h:183,w:93,isPitcher:false,wbcStats:{PA:21,AB:16,H:4,HR:1,R:7,RBI:2,BB:5,SO:5,SB:1,AVG:0.25,OBP:0.429}},
  {id:297,name:"Salvador Perez",team:"Venezuela",pool:"Pool D",pos:"C",mlbId:521692,bats:"R",throws:"R",num:13,dob:"1990-05-10",age:36,h:188,w:116,isPitcher:false,wbcStats:{PA:13,AB:12,H:3,HR:0,R:1,RBI:1,BB:1,SO:0,SB:0,AVG:0.25,OBP:0.308}},
  {id:298,name:"William Contreras",team:"Venezuela",pool:"Pool D",pos:"C",mlbId:661388,bats:"R",throws:"R",num:23,dob:"1997-12-24",age:29,h:178,w:98,isPitcher:false,wbcStats:{PA:11,AB:10,H:1,HR:0,R:1,RBI:0,BB:1,SO:1,SB:0,AVG:0.1,OBP:0.182}},
  {id:299,name:"Willson Contreras",team:"Venezuela",pool:"Pool D",pos:"1B",mlbId:575929,bats:"R",throws:"R",num:40,dob:"1992-05-13",age:34,h:183,w:109,isPitcher:false,wbcStats:{PA:12,AB:9,H:3,HR:0,R:0,RBI:2,BB:3,SO:3,SB:0,AVG:0.333,OBP:0.5}},
  {id:300,name:"Wilyer Abreu",team:"Venezuela",pool:"Pool D",pos:"OF",mlbId:677800,bats:"L",throws:"L",num:16,dob:"1999-06-24",age:27,h:178,w:98,isPitcher:false,wbcStats:{PA:20,AB:18,H:5,HR:0,R:1,RBI:3,BB:1,SO:4,SB:0,AVG:0.278,OBP:0.316}},
  {id:301,name:"Yoendrys Gomez",team:"Venezuela",pool:"Pool D",pos:"P",mlbId:672782,bats:"R",throws:"R",num:94,dob:"1999-10-15",age:27,h:191,w:96,isPitcher:true,wbcStats:{G:1,GS:1,W:1,L:0,H:0,SV:0,IP:2.0,SO:3,BB:0,HA:2,HR:0,ER:0,ERA:0.0,WHIP:1.0,Kpct:42.9,BBpct:0.0}},
  {id:302,name:"Aaron Judge",team:"USA",pool:"Pool B",pos:"RF",mlbId:592450,bats:"R",throws:"R",num:99,dob:"1992-04-26",age:34,h:201,w:128,isPitcher:false,wbcStats:{PA:27,AB:21,H:7,HR:3,R:5,RBI:8,BB:6,SO:3,SB:0,AVG:0.333,OBP:0.481}},
  {id:303,name:"Alex Bregman",team:"USA",pool:"Pool B",pos:"3B",mlbId:608324,bats:"R",throws:"R",num:2,dob:"1994-03-30",age:32,h:178,w:86,isPitcher:false,wbcStats:{PA:21,AB:12,H:3,HR:2,R:6,RBI:6,BB:6,SO:2,SB:0,AVG:0.25,OBP:0.5}},
  {id:304,name:"Bobby Witt Jr.",team:"USA",pool:"Pool B",pos:"SS",mlbId:677951,bats:"R",throws:"R",num:7,dob:"2000-06-14",age:26,h:185,w:91,isPitcher:false,wbcStats:{PA:23,AB:20,H:7,HR:0,R:3,RBI:0,BB:3,SO:2,SB:3,AVG:0.35,OBP:0.435}},
  {id:305,name:"Brad Keller",team:"USA",pool:"Pool B",pos:"P",mlbId:641745,bats:"R",throws:"R",num:40,dob:"1995-07-27",age:31,h:196,w:116,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:2.2,SO:4,BB:1,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.75,Kpct:36.4,BBpct:9.1}},
  {id:306,name:"Brice Turang",team:"USA",pool:"Pool B",pos:"2B",mlbId:668930,bats:"L",throws:"R",num:13,dob:"1999-11-21",age:27,h:183,w:86,isPitcher:false,wbcStats:{PA:17,AB:17,H:8,HR:0,R:4,RBI:6,BB:0,SO:0,SB:2,AVG:0.471,OBP:0.471}},
  {id:307,name:"Bryce Harper",team:"USA",pool:"Pool B",pos:"1B",mlbId:547180,bats:"L",throws:"R",num:24,dob:"1992-10-16",age:34,h:185,w:95,isPitcher:false,wbcStats:{PA:23,AB:21,H:5,HR:0,R:4,RBI:3,BB:1,SO:6,SB:0,AVG:0.238,OBP:0.273}},
  {id:308,name:"Byron Buxton",team:"USA",pool:"Pool B",pos:"CF",mlbId:621439,bats:"R",throws:"R",num:25,dob:"1993-12-18",age:33,h:185,w:86,isPitcher:false,wbcStats:{PA:12,AB:9,H:2,HR:1,R:3,RBI:3,BB:2,SO:2,SB:2,AVG:0.222,OBP:0.364}},
  {id:309,name:"Cal Raleigh",team:"USA",pool:"Pool B",pos:"C",mlbId:663728,bats:"S",throws:"R",num:29,dob:"1996-11-26",age:30,h:188,w:107,isPitcher:false,wbcStats:{PA:16,AB:11,H:2,HR:0,R:4,RBI:3,BB:4,SO:4,SB:0,AVG:0.182,OBP:0.4}},
  {id:310,name:"Clay Holmes",team:"USA",pool:"Pool B",pos:"P",mlbId:605280,bats:"R",throws:"R",num:35,dob:"1993-03-27",age:33,h:196,w:111,isPitcher:true,wbcStats:{G:1,GS:0,W:1,L:0,H:0,SV:0,IP:3.0,SO:6,BB:0,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.33,Kpct:66.7,BBpct:0.0}},
  {id:311,name:"Clayton Kershaw",team:"USA",pool:"Pool B",pos:"P",mlbId:477132,bats:"L",throws:"L",num:22,dob:"1988-03-19",age:38,h:193,w:102,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:0.2,SO:0,BB:1,HA:1,HR:1,ER:2,ERA:27.0,WHIP:3.0,Kpct:0.0,BBpct:25.0}},
  {id:312,name:"David Bednar",team:"USA",pool:"Pool B",pos:"P",mlbId:670280,bats:"L",throws:"R",num:53,dob:"1994-10-10",age:32,h:185,w:113,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:3.0,SO:4,BB:0,HA:2,HR:0,ER:0,ERA:0.0,WHIP:0.67,Kpct:33.3,BBpct:0.0}},
  {id:313,name:"Ernie Clement",team:"USA",pool:"Pool B",pos:"3B",mlbId:676391,bats:"R",throws:"R",num:5,dob:"1996-03-22",age:30,h:183,w:77,isPitcher:false,wbcStats:{PA:13,AB:10,H:3,HR:0,R:5,RBI:0,BB:3,SO:0,SB:0,AVG:0.3,OBP:0.462}},
  {id:314,name:"Gabe Speier",team:"USA",pool:"Pool B",pos:"P",mlbId:642100,bats:"L",throws:"L",num:55,dob:"1995-04-12",age:31,h:180,w:91,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:3.0,SO:4,BB:1,HA:2,HR:1,ER:1,ERA:3.0,WHIP:1.0,Kpct:33.3,BBpct:8.3}},
  {id:315,name:"Garrett Cleavinger",team:"USA",pool:"Pool B",pos:"P",mlbId:664076,bats:"R",throws:"L",num:60,dob:"1994-04-23",age:32,h:185,w:102,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:2.0,SO:2,BB:1,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.5,Kpct:28.6,BBpct:14.3}},
  {id:316,name:"Garrett Whitlock",team:"USA",pool:"Pool B",pos:"P",mlbId:676477,bats:"R",throws:"R",num:59,dob:"1996-06-11",age:30,h:196,w:101,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:1,IP:2.0,SO:3,BB:0,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.5,Kpct:42.9,BBpct:0.0}},
  {id:317,name:"Griffin Jax",team:"USA",pool:"Pool B",pos:"P",mlbId:643377,bats:"R",throws:"R",num:48,dob:"1994-11-22",age:32,h:188,w:88,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:2.2,SO:3,BB:0,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.38,Kpct:37.5,BBpct:0.0}},
  {id:318,name:"Gunnar Henderson",team:"USA",pool:"Pool B",pos:"SS",mlbId:683002,bats:"L",throws:"R",num:11,dob:"2001-06-29",age:25,h:191,w:100,isPitcher:false,wbcStats:{PA:17,AB:14,H:6,HR:1,R:3,RBI:5,BB:3,SO:5,SB:0,AVG:0.429,OBP:0.529}},
  {id:319,name:"Kyle Schwarber",team:"USA",pool:"Pool B",pos:"DH",mlbId:656941,bats:"L",throws:"R",num:12,dob:"1993-03-05",age:33,h:180,w:104,isPitcher:false,wbcStats:{PA:26,AB:21,H:7,HR:1,R:7,RBI:2,BB:5,SO:3,SB:0,AVG:0.333,OBP:0.462}},
  {id:320,name:"Logan Webb",team:"USA",pool:"Pool B",pos:"P",mlbId:657277,bats:"R",throws:"R",num:62,dob:"1996-11-18",age:30,h:185,w:101,isPitcher:true,wbcStats:{G:1,GS:1,W:1,L:0,H:0,SV:0,IP:4.0,SO:6,BB:0,HA:1,HR:1,ER:1,ERA:2.25,WHIP:0.25,Kpct:46.2,BBpct:0.0}},
  {id:321,name:"Mason Miller",team:"USA",pool:"Pool B",pos:"P",mlbId:695243,bats:"R",throws:"R",num:19,dob:"1998-08-24",age:28,h:196,w:91,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:3.0,SO:6,BB:1,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.33,Kpct:60.0,BBpct:10.0}},
  {id:322,name:"Matthew Boyd",team:"USA",pool:"Pool B",pos:"P",mlbId:571510,bats:"L",throws:"L",num:31,dob:"1991-02-02",age:35,h:191,w:101,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:5.0,SO:7,BB:0,HA:7,HR:2,ER:3,ERA:5.4,WHIP:1.4,Kpct:30.4,BBpct:0.0}},
  {id:323,name:"Michael Wacha",team:"USA",pool:"Pool B",pos:"P",mlbId:608379,bats:"R",throws:"R",num:52,dob:"1991-07-01",age:35,h:198,w:98,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:3.0,SO:5,BB:0,HA:4,HR:1,ER:3,ERA:9.0,WHIP:1.33,Kpct:38.5,BBpct:0.0}},
  {id:324,name:"Nolan McLean",team:"USA",pool:"Pool B",pos:"P",mlbId:690997,bats:"R",throws:"R",num:26,dob:"2001-07-24",age:25,h:188,w:97,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:1,H:0,SV:0,IP:3.0,SO:4,BB:2,HA:2,HR:2,ER:3,ERA:9.0,WHIP:1.33,Kpct:28.6,BBpct:14.3}},
  {id:325,name:"Paul Goldschmidt",team:"USA",pool:"Pool B",pos:"1B",mlbId:502671,bats:"R",throws:"R",num:43,dob:"1987-09-10",age:39,h:188,w:102,isPitcher:false,wbcStats:{PA:9,AB:7,H:3,HR:1,R:3,RBI:2,BB:1,SO:1,SB:0,AVG:0.429,OBP:0.5}},
  {id:326,name:"Paul Skenes",team:"USA",pool:"Pool B",pos:"P",mlbId:694973,bats:"R",throws:"R",num:30,dob:"2002-05-29",age:24,h:198,w:118,isPitcher:true,wbcStats:{G:2,GS:2,W:2,L:0,H:0,SV:0,IP:7.0,SO:11,BB:1,HA:2,HR:0,ER:1,ERA:1.29,WHIP:0.43,Kpct:45.8,BBpct:4.2}},
  {id:327,name:"Pete Crow Armstrong",team:"USA",pool:"Pool B",pos:"CF",mlbId:691718,bats:"L",throws:"L",num:4,dob:"2002-03-25",age:24,h:183,w:83,isPitcher:false,wbcStats:{PA:18,AB:16,H:6,HR:2,R:6,RBI:7,BB:2,SO:4,SB:3,AVG:0.375,OBP:0.444}},
  {id:328,name:"Roman Anthony",team:"USA",pool:"Pool B",pos:"RF",mlbId:701350,bats:"L",throws:"R",num:3,dob:"2004-05-13",age:22,h:191,w:91,isPitcher:false,wbcStats:{PA:24,AB:19,H:6,HR:2,R:5,RBI:8,BB:5,SO:5,SB:0,AVG:0.316,OBP:0.458}},
  {id:329,name:"Ryan Yarbrough",team:"USA",pool:"Pool B",pos:"P",mlbId:642232,bats:"R",throws:"L",num:44,dob:"1991-12-31",age:35,h:196,w:98,isPitcher:true,wbcStats:{G:2,GS:1,W:0,L:0,H:0,SV:0,IP:5.1,SO:4,BB:1,HA:4,HR:1,ER:3,ERA:5.06,WHIP:0.94,Kpct:19.0,BBpct:4.8}},
  {id:330,name:"Tarik Skubal",team:"USA",pool:"Pool B",pos:"P",mlbId:669373,bats:"R",throws:"L",num:27,dob:"1996-11-20",age:30,h:191,w:109,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:3.0,SO:5,BB:0,HA:2,HR:1,ER:1,ERA:3.0,WHIP:0.67,Kpct:45.5,BBpct:0.0}},
  {id:331,name:"Will Smith",team:"USA",pool:"Pool B",pos:"C",mlbId:669257,bats:"R",throws:"R",num:16,dob:"1995-03-28",age:31,h:178,w:88,isPitcher:false,wbcStats:{PA:13,AB:10,H:3,HR:1,R:1,RBI:2,BB:2,SO:0,SB:0,AVG:0.3,OBP:0.417}},
  {id:332,name:"Alexander Vargas",team:"Cuba",pool:"Pool A",pos:"SS",mlbId:683061,bats:"R",throws:"R",num:13,dob:"2001-10-29",age:25,h:180,w:83,isPitcher:false,wbcStats:{PA:4,AB:4,H:1,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0,AVG:0.25,OBP:0.25}},
  {id:333,name:"Alexei Ramirez",team:"Cuba",pool:"Pool A",pos:"SS",mlbId:493351,bats:"R",throws:"R",num:2,dob:"1981-09-22",age:45,h:188,w:82,isPitcher:false,wbcStats:{PA:4,AB:4,H:0,HR:0,R:0,RBI:0,BB:0,SO:3,SB:0,AVG:0.0,OBP:0.0}},
  {id:334,name:"Alfredo Despaigne",team:"Cuba",pool:"Pool A",pos:"OF",mlbId:493319,bats:"R",throws:"R",num:54,dob:"1986-06-17",age:40,h:170,w:95,isPitcher:false,wbcStats:{PA:15,AB:15,H:3,HR:0,R:0,RBI:0,BB:0,SO:2,SB:0,AVG:0.2,OBP:0.2}},
  {id:335,name:"Andrys Perez",team:"Cuba",pool:"Pool A",pos:"C",mlbId:808949,bats:"R",throws:"R",num:17,dob:"2001-02-09",age:25,h:188,w:94,isPitcher:false,wbcStats:{PA:1,AB:1,H:0,HR:0,R:0,RBI:0,BB:0,SO:1,SB:0,AVG:0.0,OBP:0.0}},
  {id:336,name:"Ariel Martinez",team:"Cuba",pool:"Pool A",pos:"OF",mlbId:673508,bats:"R",throws:"R",num:40,dob:"1996-05-28",age:30,h:188,w:89,isPitcher:false,wbcStats:{PA:18,AB:15,H:4,HR:1,R:2,RBI:3,BB:1,SO:3,SB:0,AVG:0.267,OBP:0.312}},
  {id:337,name:"Armando Duenas",team:"Cuba",pool:"Pool A",pos:"P",mlbId:807176,bats:"R",throws:"R",num:6,dob:"1994-09-17",age:32,h:191,w:98,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.2,SO:1,BB:1,HA:4,HR:1,ER:4,ERA:21.6,WHIP:3.0,Kpct:9.1,BBpct:9.1}},
  {id:338,name:"Christian Rodriguez",team:"Cuba",pool:"Pool A",pos:"IF",mlbId:839102,bats:"R",throws:"R",num:95,dob:"2002-03-31",age:24,h:183,w:73,isPitcher:false,wbcStats:{PA:1,AB:1,H:0,HR:0,R:0,RBI:0,BB:0,SO:1,SB:0,AVG:0.0,OBP:0.0}},
  {id:339,name:"Darien Nunez",team:"Cuba",pool:"Pool A",pos:"P",mlbId:628345,bats:"L",throws:"L",num:62,dob:"1993-03-19",age:33,h:188,w:93,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:2,SV:0,IP:2.1,SO:4,BB:1,HA:2,HR:0,ER:1,ERA:3.86,WHIP:1.29,Kpct:40.0,BBpct:10.0}},
  {id:340,name:"Denny Larrondo",team:"Cuba",pool:"Pool A",pos:"P",mlbId:682644,bats:"R",throws:"R",num:25,dob:"2002-05-31",age:24,h:188,w:82,isPitcher:true,wbcStats:{G:2,GS:2,W:1,L:1,H:0,SV:0,IP:4.0,SO:3,BB:5,HA:5,HR:1,ER:3,ERA:6.75,WHIP:2.5,Kpct:13.6,BBpct:22.7}},
  {id:341,name:"Emmanuel Chapman",team:"Cuba",pool:"Pool A",pos:"P",mlbId:820888,bats:"R",throws:"R",num:34,dob:"1998-09-23",age:28,h:198,w:116,isPitcher:true,wbcStats:{G:4,GS:0,W:0,L:0,H:1,SV:0,IP:4.0,SO:3,BB:3,HA:5,HR:0,ER:3,ERA:6.75,WHIP:2.0,Kpct:14.3,BBpct:14.3}},
  {id:342,name:"Erisbel Arruebarrena",team:"Cuba",pool:"Pool A",pos:"2B",mlbId:628326,bats:"R",throws:"R",num:71,dob:"1990-03-25",age:36,h:185,w:104,isPitcher:false,wbcStats:{PA:16,AB:13,H:1,HR:1,R:1,RBI:1,BB:2,SO:8,SB:0,AVG:0.077,OBP:0.2}},
  {id:343,name:"Frank Alvarez",team:"Cuba",pool:"Pool A",pos:"P",mlbId:701260,bats:"R",throws:"R",num:22,dob:"1999-01-16",age:27,h:191,w:92,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:0.1,SO:0,BB:2,HA:1,HR:0,ER:4,ERA:108.0,WHIP:9.0,Kpct:0.0,BBpct:40.0}},
  {id:344,name:"Josimar Cousin",team:"Cuba",pool:"Pool A",pos:"P",mlbId:814300,bats:"R",throws:"R",num:30,dob:"1998-02-18",age:28,h:191,w:104,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:3.2,SO:4,BB:1,HA:4,HR:1,ER:5,ERA:12.27,WHIP:1.36,Kpct:25.0,BBpct:6.2}},
  {id:345,name:"Julio Robaina",team:"Cuba",pool:"Pool A",pos:"P",mlbId:678789,bats:"L",throws:"L",num:38,dob:"2001-03-23",age:25,h:180,w:77,isPitcher:true,wbcStats:{G:2,GS:2,W:0,L:2,H:0,SV:0,IP:3.1,SO:2,BB:2,HA:8,HR:0,ER:5,ERA:13.5,WHIP:3.0,Kpct:10.0,BBpct:10.0}},
  {id:346,name:"Leonel Moas Jr.",team:"Cuba",pool:"Pool A",pos:"OF",mlbId:838354,bats:"R",throws:"R",num:20,dob:"1996-04-14",age:30,h:188,w:86,isPitcher:false,wbcStats:{PA:13,AB:13,H:2,HR:0,R:1,RBI:0,BB:0,SO:5,SB:1,AVG:0.154,OBP:0.154}},
  {id:347,name:"Livan Moinelo",team:"Cuba",pool:"Pool A",pos:"P",mlbId:661267,bats:"L",throws:"L",num:89,dob:"1995-12-08",age:31,h:178,w:70,isPitcher:true,wbcStats:{G:1,GS:1,W:1,L:0,H:0,SV:0,IP:3.2,SO:4,BB:2,HA:2,HR:0,ER:0,ERA:0.0,WHIP:1.09,Kpct:26.7,BBpct:13.3}},
  {id:348,name:"Luis Romero",team:"Cuba",pool:"Pool A",pos:"P",mlbId:673603,bats:"R",throws:"R",num:16,dob:"1994-04-23",age:32,h:183,w:92,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:2.0,SO:1,BB:2,HA:2,HR:0,ER:1,ERA:4.5,WHIP:2.0,Kpct:10.0,BBpct:20.0}},
  {id:349,name:"Malcom Nunez",team:"Cuba",pool:"Pool A",pos:"3B",mlbId:682650,bats:"R",throws:"R",num:9,dob:"2001-03-09",age:25,h:183,w:93,isPitcher:false,wbcStats:{PA:6,AB:6,H:1,HR:0,R:0,RBI:1,BB:0,SO:0,SB:0,AVG:0.167,OBP:0.167}},
  {id:350,name:"Omar Hernandez",team:"Cuba",pool:"Pool A",pos:"C",mlbId:683427,bats:"R",throws:"R",num:3,dob:"2001-12-10",age:25,h:178,w:77,isPitcher:false,wbcStats:{PA:16,AB:14,H:2,HR:0,R:0,RBI:1,BB:1,SO:4,SB:1,AVG:0.143,OBP:0.2}},
  {id:351,name:"Osiel Rodriguez",team:"Cuba",pool:"Pool A",pos:"P",mlbId:682640,bats:"R",throws:"R",num:15,dob:"2001-11-22",age:25,h:188,w:95,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:1.0,SO:0,BB:0,HA:2,HR:0,ER:1,ERA:9.0,WHIP:2.0,Kpct:0.0,BBpct:0.0}},
  {id:352,name:"Pedro Santos",team:"Cuba",pool:"Pool A",pos:"P",mlbId:683726,bats:"R",throws:"R",num:18,dob:"2000-01-07",age:26,h:193,w:93,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:3.0,SO:4,BB:5,HA:1,HR:0,ER:1,ERA:3.0,WHIP:2.0,Kpct:30.8,BBpct:38.5}},
  {id:353,name:"Raidel Martinez",team:"Cuba",pool:"Pool A",pos:"P",mlbId:673510,bats:"L",throws:"R",num:92,dob:"1996-10-11",age:30,h:193,w:93,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:2,IP:3.0,SO:3,BB:1,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.33,Kpct:30.0,BBpct:10.0}},
  {id:354,name:"Randy Martinez",team:"Cuba",pool:"Pool A",pos:"P",mlbId:838353,bats:"L",throws:"L",num:90,dob:"2003-09-28",age:23,h:175,w:72,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:1.0,SO:0,BB:1,HA:2,HR:1,ER:3,ERA:27.0,WHIP:3.0,Kpct:0.0,BBpct:16.7}},
  {id:355,name:"Roel Santos",team:"Cuba",pool:"Pool A",pos:"OF",mlbId:661270,bats:"L",throws:"L",num:1,dob:"1987-09-15",age:39,h:168,w:78,isPitcher:false,wbcStats:{PA:20,AB:16,H:2,HR:0,R:2,RBI:0,BB:3,SO:5,SB:0,AVG:0.125,OBP:0.263}},
  {id:356,name:"Yariel Rodriguez",team:"Cuba",pool:"Pool A",pos:"P",mlbId:684320,bats:"R",throws:"R",num:29,dob:"1997-03-10",age:29,h:183,w:75,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:2,SV:0,IP:6.0,SO:8,BB:1,HA:2,HR:0,ER:0,ERA:0.0,WHIP:0.5,Kpct:38.1,BBpct:4.8}},
  {id:357,name:"Yiddi Cappe",team:"Cuba",pool:"Pool A",pos:"SS",mlbId:691268,bats:"R",throws:"R",num:24,dob:"2002-09-17",age:24,h:191,w:79,isPitcher:false,wbcStats:{PA:16,AB:15,H:4,HR:0,R:1,RBI:2,BB:1,SO:2,SB:0,AVG:0.267,OBP:0.312}},
  {id:358,name:"Yoan Lopez",team:"Cuba",pool:"Pool A",pos:"P",mlbId:661255,bats:"R",throws:"R",num:35,dob:"1993-01-02",age:33,h:191,w:94,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:2,SV:0,IP:3.0,SO:0,BB:0,HA:5,HR:0,ER:0,ERA:0.0,WHIP:1.67,Kpct:0.0,BBpct:0.0}},
  {id:359,name:"Yoan Moncada",team:"Cuba",pool:"Pool A",pos:"3B",mlbId:660162,bats:"S",throws:"R",num:10,dob:"1995-05-27",age:31,h:183,w:100,isPitcher:false,wbcStats:{PA:18,AB:14,H:3,HR:1,R:3,RBI:2,BB:3,SO:6,SB:0,AVG:0.214,OBP:0.353}},
  {id:360,name:"Yoel Yanqui",team:"Cuba",pool:"Pool A",pos:"IF",mlbId:677120,bats:"L",throws:"L",num:33,dob:"1996-04-25",age:30,h:185,w:95,isPitcher:false,wbcStats:{PA:3,AB:2,H:0,HR:0,R:1,RBI:0,BB:1,SO:0,SB:0,AVG:0.0,OBP:0.333}},
  {id:361,name:"Yoelkis Guibert",team:"Cuba",pool:"Pool A",pos:"OF",mlbId:698795,bats:"L",throws:"L",num:7,dob:"1994-08-29",age:32,h:178,w:76,isPitcher:false,wbcStats:{PA:14,AB:12,H:3,HR:1,R:2,RBI:1,BB:2,SO:2,SB:0,AVG:0.25,OBP:0.357}},
  {id:362,name:"Angel Reyes",team:"Puerto Rico",pool:"Pool A",pos:"P",mlbId:681938,bats:"R",throws:"R",num:34,dob:"1997-10-17",age:29,h:188,w:93,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:2.2,SO:1,BB:1,HA:3,HR:0,ER:0,ERA:0.0,WHIP:1.5,Kpct:7.7,BBpct:7.7}},
  {id:363,name:"Bryan Torres",team:"Puerto Rico",pool:"Pool A",pos:"OF",mlbId:663494,bats:"L",throws:"R",num:2,dob:"1997-07-02",age:29,h:170,w:75,isPitcher:false,wbcStats:{PA:13,AB:10,H:4,HR:0,R:3,RBI:0,BB:3,SO:2,SB:0,AVG:0.4,OBP:0.538}},
  {id:364,name:"Carlos Cortes",team:"Puerto Rico",pool:"Pool A",pos:"LF",mlbId:666126,bats:"L",throws:"S",num:14,dob:"1997-06-30",age:29,h:170,w:89,isPitcher:false,wbcStats:{PA:20,AB:18,H:6,HR:0,R:3,RBI:1,BB:1,SO:1,SB:0,AVG:0.333,OBP:0.368}},
  {id:365,name:"Christian Vazquez",team:"Puerto Rico",pool:"Pool A",pos:"C",mlbId:543877,bats:"R",throws:"R",num:7,dob:"1990-08-21",age:36,h:173,w:93,isPitcher:false,wbcStats:{PA:13,AB:12,H:2,HR:0,R:0,RBI:0,BB:0,SO:3,SB:0,AVG:0.167,OBP:0.167}},
  {id:366,name:"Darell Hernaiz",team:"Puerto Rico",pool:"Pool A",pos:"3B",mlbId:687231,bats:"R",throws:"R",num:23,dob:"2001-08-03",age:25,h:180,w:86,isPitcher:false,wbcStats:{PA:20,AB:19,H:4,HR:1,R:2,RBI:1,BB:1,SO:2,SB:0,AVG:0.211,OBP:0.25}},
  {id:367,name:"Eddie Rosario",team:"Puerto Rico",pool:"Pool A",pos:"OF",mlbId:592696,bats:"L",throws:"R",num:17,dob:"1991-09-28",age:35,h:183,w:82,isPitcher:false,wbcStats:{PA:20,AB:16,H:4,HR:0,R:4,RBI:2,BB:4,SO:4,SB:2,AVG:0.25,OBP:0.4}},
  {id:368,name:"Eduardo Rivera",team:"Puerto Rico",pool:"Pool A",pos:"P",mlbId:700842,bats:"L",throws:"L",num:99,dob:"2003-06-13",age:23,h:201,w:108,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:4.1,SO:5,BB:1,HA:1,HR:0,ER:1,ERA:2.08,WHIP:0.46,Kpct:31.2,BBpct:6.2}},
  {id:369,name:"Edwin Arroyo",team:"Puerto Rico",pool:"Pool A",pos:"SS",mlbId:695490,bats:"S",throws:"R",num:13,dob:"2003-08-25",age:23,h:180,w:78,isPitcher:false,wbcStats:{PA:6,AB:6,H:2,HR:0,R:1,RBI:1,BB:0,SO:1,SB:0,AVG:0.333,OBP:0.333}},
  {id:370,name:"Edwin Diaz",team:"Puerto Rico",pool:"Pool A",pos:"P",mlbId:621242,bats:"R",throws:"R",num:39,dob:"1994-03-22",age:32,h:191,w:75,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:1,IP:2.0,SO:5,BB:0,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.5,Kpct:71.4,BBpct:0.0}},
  {id:371,name:"Elmer Rodriguez",team:"Puerto Rico",pool:"Pool A",pos:"P",mlbId:695684,bats:"L",throws:"R",num:18,dob:"2003-08-18",age:23,h:191,w:73,isPitcher:true,wbcStats:{G:2,GS:2,W:2,L:0,H:0,SV:0,IP:6.0,SO:5,BB:5,HA:2,HR:0,ER:0,ERA:0.0,WHIP:1.17,Kpct:20.8,BBpct:20.8}},
  {id:372,name:"Emmanuel Rivera",team:"Puerto Rico",pool:"Pool A",pos:"3B",mlbId:656896,bats:"R",throws:"R",num:26,dob:"1996-06-29",age:30,h:183,w:102,isPitcher:false,wbcStats:{PA:19,AB:19,H:4,HR:0,R:2,RBI:1,BB:0,SO:1,SB:0,AVG:0.211,OBP:0.211}},
  {id:373,name:"Fernando Cruz",team:"Puerto Rico",pool:"Pool A",pos:"P",mlbId:518585,bats:"R",throws:"R",num:63,dob:"1990-03-28",age:36,h:188,w:108,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:2,SV:0,IP:1.1,SO:2,BB:1,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.75,Kpct:40.0,BBpct:20.0}},
  {id:374,name:"Gabriel Rodriguez",team:"Puerto Rico",pool:"Pool A",pos:"P",mlbId:676082,bats:"L",throws:"L",num:69,dob:"1999-04-09",age:27,h:185,w:88,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:1.0,SO:2,BB:1,HA:1,HR:0,ER:1,ERA:9.0,WHIP:2.0,Kpct:40.0,BBpct:20.0}},
  {id:375,name:"Heliot Ramos",team:"Puerto Rico",pool:"Pool A",pos:"OF",mlbId:671218,bats:"R",throws:"R",num:22,dob:"1999-09-07",age:27,h:180,w:104,isPitcher:false,wbcStats:{PA:23,AB:19,H:2,HR:0,R:2,RBI:1,BB:2,SO:7,SB:0,AVG:0.105,OBP:0.19}},
  {id:376,name:"Jorge Lopez",team:"Puerto Rico",pool:"Pool A",pos:"P",mlbId:605347,bats:"R",throws:"R",num:48,dob:"1993-02-10",age:33,h:191,w:91,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.2,SO:1,BB:1,HA:4,HR:0,ER:2,ERA:10.8,WHIP:3.0,Kpct:9.1,BBpct:9.1}},
  {id:377,name:"Jose De Leon",team:"Puerto Rico",pool:"Pool A",pos:"P",mlbId:592254,bats:"R",throws:"R",num:87,dob:"1992-08-07",age:34,h:183,w:100,isPitcher:true,wbcStats:{G:2,GS:1,W:0,L:1,H:0,SV:0,IP:3.2,SO:2,BB:1,HA:2,HR:0,ER:2,ERA:4.91,WHIP:0.82,Kpct:13.3,BBpct:6.7}},
  {id:378,name:"Jose Espada",team:"Puerto Rico",pool:"Pool A",pos:"P",mlbId:664744,bats:"R",throws:"R",num:35,dob:"1997-02-22",age:29,h:183,w:77,isPitcher:true,wbcStats:{G:3,GS:0,W:1,L:0,H:0,SV:0,IP:3.1,SO:3,BB:3,HA:3,HR:0,ER:2,ERA:5.4,WHIP:1.8,Kpct:20.0,BBpct:20.0}},
  {id:379,name:"Jovani Moran",team:"Puerto Rico",pool:"Pool A",pos:"P",mlbId:663558,bats:"L",throws:"L",num:16,dob:"1997-04-24",age:29,h:185,w:76,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:4.0,SO:2,BB:2,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.5,Kpct:14.3,BBpct:14.3}},
  {id:380,name:"Luis Quinones",team:"Puerto Rico",pool:"Pool A",pos:"P",mlbId:687879,bats:"R",throws:"R",num:51,dob:"1997-07-02",age:29,h:183,w:93,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:5.1,SO:9,BB:3,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.56,Kpct:50.0,BBpct:16.7}},
  {id:381,name:"Luis Vazquez",team:"Puerto Rico",pool:"Pool A",pos:"SS",mlbId:676679,bats:"R",throws:"R",num:8,dob:"1999-10-10",age:27,h:183,w:75,isPitcher:false,wbcStats:{PA:8,AB:7,H:1,HR:0,R:0,RBI:0,BB:0,SO:3,SB:0,AVG:0.143,OBP:0.143}},
  {id:382,name:"MJ Melendez",team:"Puerto Rico",pool:"Pool A",pos:"LF",mlbId:669004,bats:"L",throws:"R",num:10,dob:"1998-11-29",age:28,h:183,w:86,isPitcher:false,wbcStats:{PA:11,AB:7,H:0,HR:0,R:1,RBI:1,BB:3,SO:3,SB:0,AVG:0.0,OBP:0.3}},
  {id:383,name:"Martin Maldonado",team:"Puerto Rico",pool:"Pool A",pos:"C",mlbId:455117,bats:"R",throws:"R",num:15,dob:"1986-08-16",age:40,h:183,w:104,isPitcher:false,wbcStats:{PA:15,AB:12,H:3,HR:0,R:2,RBI:5,BB:1,SO:4,SB:0,AVG:0.25,OBP:0.308}},
  {id:384,name:"Matthew Lugo",team:"Puerto Rico",pool:"Pool A",pos:"LF",mlbId:683090,bats:"R",throws:"R",num:6,dob:"2001-05-09",age:25,h:183,w:86,isPitcher:false,wbcStats:{PA:12,AB:9,H:1,HR:0,R:0,RBI:0,BB:3,SO:3,SB:1,AVG:0.111,OBP:0.333}},
  {id:385,name:"Nolan Arenado",team:"Puerto Rico",pool:"Pool A",pos:"3B",mlbId:571448,bats:"R",throws:"R",num:28,dob:"1991-04-16",age:35,h:188,w:98,isPitcher:false,wbcStats:{PA:21,AB:18,H:4,HR:0,R:2,RBI:3,BB:1,SO:3,SB:0,AVG:0.222,OBP:0.263}},
  {id:386,name:"Raymond Burgos",team:"Puerto Rico",pool:"Pool A",pos:"P",mlbId:670103,bats:"L",throws:"L",num:29,dob:"1998-11-29",age:28,h:196,w:95,isPitcher:true,wbcStats:{G:2,GS:1,W:0,L:1,H:0,SV:0,IP:4.0,SO:4,BB:1,HA:5,HR:0,ER:3,ERA:6.75,WHIP:1.5,Kpct:25.0,BBpct:6.2}},
  {id:387,name:"Ricardo Velez",team:"Puerto Rico",pool:"Pool A",pos:"P",mlbId:703106,bats:"R",throws:"R",num:44,dob:"1998-08-21",age:28,h:185,w:82,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:2.1,SO:1,BB:2,HA:4,HR:0,ER:1,ERA:3.86,WHIP:2.57,Kpct:7.1,BBpct:14.3}},
  {id:388,name:"Rico Garcia",team:"Puerto Rico",pool:"Pool A",pos:"P",mlbId:670329,bats:"R",throws:"R",num:52,dob:"1994-01-10",age:32,h:175,w:91,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:1.2,SO:2,BB:5,HA:2,HR:0,ER:1,ERA:5.4,WHIP:4.2,Kpct:16.7,BBpct:41.7}},
  {id:389,name:"Seth Lugo",team:"Puerto Rico",pool:"Pool A",pos:"P",mlbId:607625,bats:"R",throws:"R",num:67,dob:"1989-11-17",age:37,h:193,w:102,isPitcher:true,wbcStats:{G:1,GS:1,W:1,L:0,H:0,SV:0,IP:4.0,SO:3,BB:2,HA:3,HR:0,ER:0,ERA:0.0,WHIP:1.25,Kpct:17.6,BBpct:11.8}},
  {id:390,name:"Willi Castro",team:"Puerto Rico",pool:"Pool A",pos:"LF",mlbId:650489,bats:"S",throws:"R",num:3,dob:"1997-04-24",age:29,h:183,w:93,isPitcher:false,wbcStats:{PA:20,AB:16,H:5,HR:0,R:1,RBI:2,BB:4,SO:3,SB:1,AVG:0.312,OBP:0.45}},
  {id:391,name:"Yacksel Rios",team:"Puerto Rico",pool:"Pool A",pos:"P",mlbId:605441,bats:"R",throws:"R",num:75,dob:"1993-06-27",age:33,h:191,w:98,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:1,IP:4.2,SO:5,BB:2,HA:2,HR:0,ER:0,ERA:0.0,WHIP:0.86,Kpct:27.8,BBpct:11.1}},
  {id:392,name:"Bo Takahashi",team:"Brazil",pool:"Pool B",pos:"P",mlbId:649963,bats:"R",throws:"R",num:26,dob:"1997-01-23",age:29,h:183,w:102,isPitcher:true,wbcStats:{G:3,GS:2,W:0,L:2,H:0,SV:0,IP:3.0,SO:2,BB:5,HA:8,HR:3,ER:9,ERA:27.0,WHIP:4.33,Kpct:8.3,BBpct:20.8}},
  {id:393,name:"Caio De Araujo",team:"Brazil",pool:"Pool B",pos:"P",mlbId:822885,bats:"R",throws:"R",num:8,dob:"2002-02-07",age:24,h:188,w:102,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:1.0,SO:1,BB:1,HA:3,HR:1,ER:2,ERA:18.0,WHIP:4.0,Kpct:14.3,BBpct:14.3}},
  {id:394,name:"Dante Bichette Jr.",team:"Brazil",pool:"Pool B",pos:"1B",mlbId:605142,bats:"R",throws:"R",num:77,dob:"1992-09-26",age:34,h:188,w:93,isPitcher:false,wbcStats:{PA:8,AB:7,H:0,HR:0,R:0,RBI:0,BB:1,SO:4,SB:0,AVG:0.0,OBP:0.125}},
  {id:395,name:"Enzo Hayashida",team:"Brazil",pool:"Pool B",pos:"C",mlbId:838357,bats:"R",throws:"R",num:20,dob:"2008-01-01",age:18,h:173,w:82,isPitcher:false,wbcStats:{PA:7,AB:7,H:1,HR:0,R:0,RBI:0,BB:0,SO:2,SB:0,AVG:0.143,OBP:0.143}},
  {id:396,name:"Enzo Sawayama",team:"Brazil",pool:"Pool B",pos:"P",mlbId:807344,bats:"L",throws:"L",num:18,dob:"2003-10-15",age:23,h:183,w:100,isPitcher:true,wbcStats:{G:3,GS:3,W:0,L:0,H:0,SV:0,IP:10.0,SO:8,BB:2,HA:6,HR:1,ER:1,ERA:0.9,WHIP:0.8,Kpct:21.6,BBpct:5.4}},
  {id:397,name:"Eric Pardinho",team:"Brazil",pool:"Pool B",pos:"P",mlbId:672078,bats:"R",throws:"R",num:43,dob:"2001-01-05",age:25,h:178,w:70,isPitcher:true,wbcStats:{G:2,GS:1,W:0,L:1,H:0,SV:0,IP:4.0,SO:2,BB:2,HA:12,HR:1,ER:12,ERA:27.0,WHIP:3.5,Kpct:7.7,BBpct:7.7}},
  {id:398,name:"Felipe Koragi",team:"Brazil",pool:"Pool B",pos:"2B",mlbId:806889,bats:"R",throws:"R",num:2,dob:"2004-01-23",age:22,h:170,w:68,isPitcher:false,wbcStats:{PA:4,AB:3,H:1,HR:0,R:0,RBI:0,BB:0,SO:1,SB:0,AVG:0.333,OBP:0.333}},
  {id:399,name:"Felipe Mizukosi",team:"Brazil",pool:"Pool B",pos:"SS",mlbId:831113,bats:"R",throws:"R",num:39,dob:"1994-11-26",age:32,h:175,w:85,isPitcher:false,wbcStats:{PA:3,AB:2,H:1,HR:0,R:1,RBI:0,BB:1,SO:1,SB:0,AVG:0.5,OBP:0.667}},
  {id:400,name:"Gabriel Barbosa",team:"Brazil",pool:"Pool B",pos:"P",mlbId:682858,bats:"R",throws:"R",num:31,dob:"2002-01-22",age:24,h:183,w:83,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.1,SO:1,BB:3,HA:3,HR:1,ER:5,ERA:33.75,WHIP:4.5,Kpct:9.1,BBpct:27.3}},
  {id:401,name:"Gabriel Carmo",team:"Brazil",pool:"Pool B",pos:"C",mlbId:693861,bats:"R",throws:"R",num:7,dob:"1995-05-17",age:31,h:180,w:79,isPitcher:false,wbcStats:{PA:13,AB:12,H:3,HR:0,R:0,RBI:1,BB:1,SO:4,SB:0,AVG:0.25,OBP:0.308}},
  {id:402,name:"Gabriel Gomes",team:"Brazil",pool:"Pool B",pos:"C",mlbId:806238,bats:"R",throws:"R",num:28,dob:"2004-03-31",age:22,h:175,w:77,isPitcher:false,wbcStats:{PA:11,AB:10,H:2,HR:0,R:0,RBI:2,BB:1,SO:6,SB:0,AVG:0.2,OBP:0.273}},
  {id:403,name:"Gabriel Maciel",team:"Brazil",pool:"Pool B",pos:"CF",mlbId:670622,bats:"S",throws:"R",num:3,dob:"1999-01-10",age:27,h:173,w:84,isPitcher:false,wbcStats:{PA:17,AB:15,H:5,HR:0,R:1,RBI:0,BB:0,SO:5,SB:0,AVG:0.333,OBP:0.333}},
  {id:404,name:"Hector Villarroel",team:"Brazil",pool:"Pool B",pos:"P",mlbId:622609,bats:"L",throws:"L",num:4,dob:"1995-08-12",age:31,h:196,w:87,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:1,H:0,SV:0,IP:4.1,SO:1,BB:3,HA:4,HR:0,ER:2,ERA:4.15,WHIP:1.62,Kpct:5.0,BBpct:15.0}},
  {id:405,name:"Hugo Kanabushi",team:"Brazil",pool:"Pool B",pos:"P",mlbId:627387,bats:"L",throws:"L",num:66,dob:"1989-05-22",age:37,h:180,w:82,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.2,SO:0,BB:3,HA:6,HR:2,ER:5,ERA:27.0,WHIP:5.4,Kpct:0.0,BBpct:21.4}},
  {id:406,name:"Joao Gabriel Marostica",team:"Brazil",pool:"Pool B",pos:"P",mlbId:806891,bats:"R",throws:"R",num:0,dob:"2004-11-08",age:22,h:191,w:92,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:5.0,SO:0,BB:7,HA:4,HR:1,ER:4,ERA:7.2,WHIP:2.2,Kpct:0.0,BBpct:25.9}},
  {id:407,name:"Joseph Contreras",team:"Brazil",pool:"Pool B",pos:"P",mlbId:828631,bats:"S",throws:"R",num:21,dob:"2008-05-06",age:18,h:193,w:88,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:2.2,SO:2,BB:6,HA:4,HR:0,ER:3,ERA:10.12,WHIP:3.75,Kpct:12.5,BBpct:37.5}},
  {id:408,name:"Leonardo Reginatto",team:"Brazil",pool:"Pool B",pos:"SS",mlbId:553891,bats:"R",throws:"R",num:5,dob:"1990-04-10",age:36,h:188,w:85,isPitcher:false,wbcStats:{PA:20,AB:17,H:5,HR:0,R:1,RBI:1,BB:2,SO:3,SB:0,AVG:0.294,OBP:0.368}},
  {id:409,name:"Lucas Ramirez",team:"Brazil",pool:"Pool B",pos:"OF",mlbId:813429,bats:"L",throws:"R",num:24,dob:"2006-01-16",age:20,h:191,w:93,isPitcher:false,wbcStats:{PA:23,AB:19,H:4,HR:3,R:4,RBI:3,BB:3,SO:6,SB:0,AVG:0.211,OBP:0.318}},
  {id:410,name:"Lucas Rojo",team:"Brazil",pool:"Pool B",pos:"3B",mlbId:624531,bats:"R",throws:"R",num:15,dob:"1994-04-05",age:32,h:168,w:68,isPitcher:false,wbcStats:{PA:20,AB:19,H:2,HR:1,R:2,RBI:3,BB:0,SO:6,SB:0,AVG:0.105,OBP:0.105}},
  {id:411,name:"Matheus Silva",team:"Brazil",pool:"Pool B",pos:"C",mlbId:683844,bats:"R",throws:"R",num:13,dob:"2002-06-12",age:24,h:170,w:94,isPitcher:false,wbcStats:{PA:2,AB:2,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0,AVG:0.0,OBP:0.0}},
  {id:412,name:"Murilo Gouvea",team:"Brazil",pool:"Pool B",pos:"P",mlbId:517209,bats:"R",throws:"R",num:34,dob:"1988-09-15",age:38,h:191,w:95,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.2,SO:3,BB:3,HA:5,HR:1,ER:6,ERA:32.4,WHIP:4.8,Kpct:23.1,BBpct:23.1}},
  {id:413,name:"Oscar Nakaoshi",team:"Brazil",pool:"Pool B",pos:"P",mlbId:627393,bats:"L",throws:"L",num:41,dob:"1991-03-28",age:35,h:180,w:83,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.1,SO:0,BB:2,HA:5,HR:1,ER:3,ERA:20.25,WHIP:5.25,Kpct:0.0,BBpct:18.2}},
  {id:414,name:"Osvaldo Carvalho",team:"Brazil",pool:"Pool B",pos:"OF",mlbId:831112,bats:"L",throws:"R",num:6,dob:"2001-06-08",age:25,h:185,w:95,isPitcher:false,wbcStats:{PA:20,AB:19,H:4,HR:0,R:0,RBI:0,BB:1,SO:7,SB:0,AVG:0.211,OBP:0.25}},
  {id:415,name:"Thyago Vieira",team:"Brazil",pool:"Pool B",pos:"P",mlbId:600986,bats:"R",throws:"R",num:49,dob:"1993-01-07",age:33,h:191,w:117,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:0.1,SO:0,BB:2,HA:0,HR:0,ER:2,ERA:54.0,WHIP:6.0,Kpct:0.0,BBpct:66.7}},
  {id:416,name:"Tiago Da Silva",team:"Brazil",pool:"Pool B",pos:"P",mlbId:547921,bats:"R",throws:"R",num:22,dob:"1985-03-28",age:41,h:175,w:82,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:1,H:0,SV:0,IP:4.0,SO:2,BB:3,HA:3,HR:1,ER:3,ERA:6.75,WHIP:1.5,Kpct:11.1,BBpct:16.7}},
  {id:417,name:"Tiago Nishiyama",team:"Brazil",pool:"Pool B",pos:"2B",mlbId:831114,bats:"L",throws:"R",num:12,dob:"2005-11-20",age:21,h:178,w:83,isPitcher:false,wbcStats:{PA:1,AB:1,H:0,HR:0,R:0,RBI:0,BB:0,SO:1,SB:0,AVG:0.0,OBP:0.0}},
  {id:418,name:"Tomas Lopez",team:"Brazil",pool:"Pool B",pos:"P",mlbId:838355,bats:"R",throws:"R",num:35,dob:"2004-12-10",age:22,h:193,w:98,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:0.1,SO:1,BB:3,HA:1,HR:0,ER:4,ERA:108.0,WHIP:12.0,Kpct:20.0,BBpct:60.0}},
  {id:419,name:"Victor Mascai",team:"Brazil",pool:"Pool B",pos:"OF",mlbId:678685,bats:"L",throws:"R",num:17,dob:"2001-02-10",age:25,h:188,w:99,isPitcher:false,wbcStats:{PA:18,AB:15,H:3,HR:1,R:2,RBI:2,BB:3,SO:9,SB:0,AVG:0.2,OBP:0.333}},
  {id:420,name:"Vitor Ito",team:"Brazil",pool:"Pool B",pos:"SS",mlbId:672074,bats:"L",throws:"R",num:1,dob:"1995-02-16",age:31,h:178,w:81,isPitcher:false,wbcStats:{PA:20,AB:17,H:3,HR:0,R:1,RBI:0,BB:1,SO:5,SB:0,AVG:0.176,OBP:0.222}},
  {id:421,name:"Vitor Takahashi",team:"Brazil",pool:"Pool B",pos:"P",mlbId:836637,bats:"R",throws:"R",num:40,dob:"2008-03-11",age:18,h:183,w:96,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:1,H:0,SV:0,IP:2.0,SO:0,BB:4,HA:5,HR:2,ER:9,ERA:40.5,WHIP:4.5,Kpct:0.0,BBpct:25.0}},
  {id:422,name:"Adrian Almeida",team:"Colombia",pool:"Pool A",pos:"P",mlbId:622426,bats:"L",throws:"L",num:19,dob:"1995-02-25",age:31,h:183,w:73,isPitcher:true,wbcStats:{G:3,GS:1,W:0,L:1,H:0,SV:0,IP:6.1,SO:6,BB:0,HA:4,HR:0,ER:3,ERA:4.26,WHIP:0.63,Kpct:23.1,BBpct:0.0}},
  {id:423,name:"Austin Bergner",team:"Colombia",pool:"Pool A",pos:"P",mlbId:666123,bats:"R",throws:"R",num:45,dob:"1997-05-01",age:29,h:196,w:95,isPitcher:true,wbcStats:{G:3,GS:1,W:1,L:1,H:0,SV:0,IP:4.0,SO:3,BB:2,HA:4,HR:1,ER:3,ERA:6.75,WHIP:1.5,Kpct:16.7,BBpct:11.1}},
  {id:424,name:"Brayan Buelvas",team:"Colombia",pool:"Pool A",pos:"OF",mlbId:683454,bats:"R",throws:"R",num:12,dob:"2002-06-08",age:24,h:178,w:70,isPitcher:false,wbcStats:{PA:11,AB:8,H:1,HR:0,R:0,RBI:0,BB:3,SO:5,SB:0,AVG:0.125,OBP:0.364}},
  {id:425,name:"Carlos Arroyo",team:"Colombia",pool:"Pool A",pos:"2B",mlbId:678863,bats:"R",throws:"R",num:40,dob:"2001-07-11",age:25,h:175,w:77,isPitcher:false,wbcStats:{PA:5,AB:4,H:0,HR:0,R:1,RBI:0,BB:1,SO:3,SB:0,AVG:0.0,OBP:0.2}},
  {id:426,name:"Carlos Martinez",team:"Colombia",pool:"Pool A",pos:"C",mlbId:642751,bats:"R",throws:"R",num:70,dob:"1995-05-02",age:31,h:180,w:88,isPitcher:false,wbcStats:{PA:8,AB:8,H:1,HR:0,R:0,RBI:0,BB:0,SO:1,SB:0,AVG:0.125,OBP:0.125}},
  {id:427,name:"Daniel Vellojin",team:"Colombia",pool:"Pool A",pos:"C",mlbId:678962,bats:"L",throws:"R",num:51,dob:"2000-03-15",age:26,h:180,w:73,isPitcher:false,wbcStats:{PA:12,AB:11,H:1,HR:0,R:0,RBI:1,BB:0,SO:3,SB:0,AVG:0.091,OBP:0.091}},
  {id:428,name:"Danis Correa",team:"Colombia",pool:"Pool A",pos:"P",mlbId:672550,bats:"R",throws:"R",num:41,dob:"1999-08-26",age:27,h:180,w:68,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:1,H:1,SV:0,IP:2.0,SO:4,BB:0,HA:3,HR:0,ER:2,ERA:9.0,WHIP:1.5,Kpct:44.4,BBpct:0.0}},
  {id:429,name:"David Lorduy",team:"Colombia",pool:"Pool A",pos:"P",mlbId:800124,bats:"R",throws:"R",num:39,dob:"2003-10-15",age:23,h:183,w:90,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:4.1,SO:1,BB:0,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.23,Kpct:7.1,BBpct:0.0}},
  {id:430,name:"Dayan Frias",team:"Colombia",pool:"Pool A",pos:"3B",mlbId:682679,bats:"S",throws:"R",num:71,dob:"2002-06-25",age:24,h:175,w:64,isPitcher:false,wbcStats:{PA:17,AB:15,H:3,HR:0,R:0,RBI:0,BB:2,SO:2,SB:1,AVG:0.2,OBP:0.294}},
  {id:431,name:"Donovan Solano",team:"Colombia",pool:"Pool A",pos:"1B",mlbId:456781,bats:"R",throws:"R",num:7,dob:"1987-12-17",age:39,h:173,w:95,isPitcher:false,wbcStats:{PA:24,AB:15,H:1,HR:0,R:2,RBI:0,BB:9,SO:6,SB:0,AVG:0.067,OBP:0.417}},
  {id:432,name:"Elkin Alcala",team:"Colombia",pool:"Pool A",pos:"P",mlbId:666669,bats:"R",throws:"R",num:26,dob:"1997-08-02",age:29,h:180,w:79,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:2.1,SO:3,BB:0,HA:2,HR:0,ER:0,ERA:0.0,WHIP:0.86,Kpct:33.3,BBpct:0.0}},
  {id:433,name:"Emerson Martinez",team:"Colombia",pool:"Pool A",pos:"P",mlbId:627218,bats:"R",throws:"R",num:96,dob:"1995-01-11",age:31,h:183,w:86,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:4.1,SO:4,BB:1,HA:4,HR:0,ER:0,ERA:0.0,WHIP:1.15,Kpct:22.2,BBpct:5.6}},
  {id:434,name:"Ezequiel Zabaleta",team:"Colombia",pool:"Pool A",pos:"P",mlbId:664375,bats:"R",throws:"R",num:10,dob:"1995-08-20",age:31,h:183,w:79,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:4.0,SO:5,BB:2,HA:4,HR:0,ER:0,ERA:0.0,WHIP:1.5,Kpct:26.3,BBpct:10.5}},
  {id:435,name:"Gio Urshela",team:"Colombia",pool:"Pool A",pos:"3B",mlbId:570482,bats:"R",throws:"R",num:29,dob:"1991-10-11",age:35,h:183,w:98,isPitcher:false,wbcStats:{PA:18,AB:12,H:2,HR:0,R:1,RBI:1,BB:5,SO:6,SB:0,AVG:0.167,OBP:0.412}},
  {id:436,name:"Guillo Zuniga",team:"Colombia",pool:"Pool A",pos:"P",mlbId:670871,bats:"R",throws:"R",num:66,dob:"1998-10-10",age:28,h:196,w:104,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:1.1,SO:0,BB:2,HA:8,HR:0,ER:7,ERA:47.25,WHIP:7.5,Kpct:0.0,BBpct:15.4}},
  {id:437,name:"Gustavo Campero",team:"Colombia",pool:"Pool A",pos:"OF",mlbId:672569,bats:"S",throws:"R",num:57,dob:"1997-09-20",age:29,h:168,w:83,isPitcher:false,wbcStats:{PA:17,AB:15,H:1,HR:0,R:0,RBI:0,BB:2,SO:4,SB:0,AVG:0.067,OBP:0.176}},
  {id:438,name:"Harold Ramirez",team:"Colombia",pool:"Pool A",pos:"OF",mlbId:623912,bats:"R",throws:"R",num:43,dob:"1994-09-06",age:32,h:180,w:105,isPitcher:false,wbcStats:{PA:22,AB:22,H:7,HR:0,R:1,RBI:1,BB:0,SO:3,SB:0,AVG:0.318,OBP:0.318}},
  {id:439,name:"Jesus Marriaga",team:"Colombia",pool:"Pool A",pos:"OF",mlbId:667354,bats:"R",throws:"R",num:16,dob:"1998-12-17",age:28,h:183,w:77,isPitcher:false,wbcStats:{PA:9,AB:8,H:2,HR:0,R:1,RBI:0,BB:1,SO:3,SB:0,AVG:0.25,OBP:0.333}},
  {id:440,name:"Jhon Romero",team:"Colombia",pool:"Pool A",pos:"P",mlbId:664584,bats:"R",throws:"R",num:73,dob:"1995-01-17",age:31,h:178,w:88,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:2.0,SO:2,BB:3,HA:4,HR:0,ER:1,ERA:4.5,WHIP:3.5,Kpct:14.3,BBpct:21.4}},
  {id:441,name:"Jordan Diaz",team:"Colombia",pool:"Pool A",pos:"2B",mlbId:672478,bats:"R",throws:"R",num:13,dob:"2000-08-13",age:26,h:175,w:79,isPitcher:false,wbcStats:{PA:17,AB:13,H:2,HR:0,R:0,RBI:2,BB:2,SO:3,SB:0,AVG:0.154,OBP:0.267}},
  {id:442,name:"Jose Quintana",team:"Colombia",pool:"Pool A",pos:"P",mlbId:500779,bats:"R",throws:"L",num:62,dob:"1989-01-24",age:37,h:188,w:102,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:3.0,SO:1,BB:1,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.33,Kpct:11.1,BBpct:11.1}},
  {id:443,name:"Julio Teheran",team:"Colombia",pool:"Pool A",pos:"P",mlbId:527054,bats:"R",throws:"R",num:49,dob:"1991-01-27",age:35,h:188,w:93,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:1.0,SO:0,BB:0,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.0,Kpct:0.0,BBpct:0.0}},
  {id:444,name:"Luis Escobar",team:"Colombia",pool:"Pool A",pos:"P",mlbId:650813,bats:"R",throws:"R",num:78,dob:"1996-05-30",age:30,h:185,w:93,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.2,SO:0,BB:2,HA:0,HR:0,ER:0,ERA:0.0,WHIP:1.2,Kpct:0.0,BBpct:28.6}},
  {id:445,name:"Luis Patino",team:"Colombia",pool:"Pool A",pos:"P",mlbId:672715,bats:"R",throws:"R",num:77,dob:"1999-10-26",age:27,h:185,w:87,isPitcher:true,wbcStats:{G:2,GS:2,W:0,L:2,H:0,SV:0,IP:1.0,SO:1,BB:3,HA:4,HR:2,ER:5,ERA:45.0,WHIP:7.0,Kpct:9.1,BBpct:27.3}},
  {id:446,name:"Michael Arroyo",team:"Colombia",pool:"Pool A",pos:"SS",mlbId:703197,bats:"R",throws:"R",num:8,dob:"2004-11-03",age:22,h:178,w:73,isPitcher:false,wbcStats:{PA:25,AB:18,H:5,HR:0,R:4,RBI:2,BB:4,SO:4,SB:2,AVG:0.278,OBP:0.409}},
  {id:447,name:"Pedro Garcia",team:"Colombia",pool:"Pool A",pos:"P",mlbId:690345,bats:"R",throws:"R",num:72,dob:"1995-03-21",age:31,h:180,w:100,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:1,IP:2.0,SO:2,BB:2,HA:5,HR:2,ER:4,ERA:18.0,WHIP:3.5,Kpct:14.3,BBpct:14.3}},
  {id:448,name:"Reynaldo Rodriguez",team:"Colombia",pool:"Pool A",pos:"1B",mlbId:471906,bats:"R",throws:"R",num:17,dob:"1986-07-02",age:40,h:183,w:88,isPitcher:false,wbcStats:{PA:17,AB:17,H:2,HR:0,R:1,RBI:2,BB:0,SO:7,SB:0,AVG:0.118,OBP:0.118}},
  {id:449,name:"Rio Gomez",team:"Colombia",pool:"Pool A",pos:"P",mlbId:677013,bats:"L",throws:"L",num:92,dob:"1994-10-20",age:32,h:183,w:86,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:2.1,SO:2,BB:1,HA:3,HR:1,ER:2,ERA:7.71,WHIP:1.71,Kpct:16.7,BBpct:8.3}},
  {id:450,name:"Tito Polo",team:"Colombia",pool:"Pool A",pos:"OF",mlbId:622738,bats:"R",throws:"R",num:23,dob:"1994-08-23",age:32,h:178,w:88,isPitcher:false,wbcStats:{PA:10,AB:8,H:2,HR:0,R:1,RBI:2,BB:0,SO:4,SB:0,AVG:0.25,OBP:0.25}},
  {id:451,name:"Yapson Gomez",team:"Colombia",pool:"Pool A",pos:"P",mlbId:642857,bats:"L",throws:"L",num:30,dob:"1993-10-02",age:33,h:178,w:73,isPitcher:true,wbcStats:{G:4,GS:0,W:0,L:0,H:1,SV:0,IP:3.0,SO:3,BB:1,HA:2,HR:0,ER:1,ERA:3.0,WHIP:1.0,Kpct:23.1,BBpct:7.7}},
  {id:452,name:"Boris Vecerka",team:"Czechia",pool:"Pool C",pos:"P",mlbId:807828,bats:"R",throws:"R",num:54,dob:"2003-08-22",age:23,h:191,w:93,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:0.0,SO:0,BB:1,HA:1,HR:0,ER:2,Kpct:0.0,BBpct:33.3}},
  {id:453,name:"Daniel Padysak",team:"Czechia",pool:"Pool C",pos:"P",mlbId:804607,bats:"R",throws:"R",num:42,dob:"2000-07-02",age:26,h:196,w:104,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:1,H:0,SV:0,IP:0.1,SO:0,BB:2,HA:2,HR:1,ER:4,ERA:108.0,WHIP:12.0,Kpct:0.0,BBpct:40.0}},
  {id:454,name:"Filip Capka",team:"Czechia",pool:"Pool C",pos:"P",mlbId:693972,bats:"R",throws:"R",num:14,dob:"1998-11-04",age:28,h:188,w:92,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:2.0,SO:1,BB:0,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.5,Kpct:14.3,BBpct:0.0}},
  {id:455,name:"Filip Kollmann",team:"Czechia",pool:"Pool C",pos:"P",mlbId:831793,bats:"L",throws:"R",num:26,dob:"2005-06-22",age:21,h:185,w:66,isPitcher:true,wbcStats:{G:0,GS:0,W:0,L:0,H:0,SV:0,IP:0.0,SO:0,BB:0,HA:0,HR:0,ER:0}},
  {id:456,name:"Jan Novak",team:"Czechia",pool:"Pool C",pos:"P",mlbId:626416,bats:"R",throws:"L",num:18,dob:"1994-01-19",age:32,h:188,w:87,isPitcher:true,wbcStats:{G:3,GS:1,W:0,L:1,H:0,SV:0,IP:3.2,SO:3,BB:4,HA:7,HR:1,ER:9,ERA:22.09,WHIP:3.0,Kpct:13.6,BBpct:18.2}},
  {id:457,name:"Jan Pospisil",team:"Czechia",pool:"Pool C",pos:"3B",mlbId:837160,bats:"L",throws:"R",num:27,dob:"2003-10-21",age:23,h:183,w:91,isPitcher:false,wbcStats:{PA:5,AB:4,H:0,HR:0,R:0,RBI:1,BB:0,SO:1,SB:0,AVG:0.0,OBP:0.0}},
  {id:458,name:"Jeff Barto",team:"Czechia",pool:"Pool C",pos:"P",mlbId:808951,bats:"L",throws:"L",num:19,dob:"1989-09-15",age:37,h:183,w:86,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:3.1,SO:1,BB:2,HA:5,HR:1,ER:3,ERA:8.1,WHIP:2.1,Kpct:5.9,BBpct:11.8}},
  {id:459,name:"Lukas Ercoli",team:"Czechia",pool:"Pool C",pos:"P",mlbId:807155,bats:"R",throws:"L",num:63,dob:"1996-04-17",age:30,h:191,w:91,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:1.2,SO:0,BB:0,HA:2,HR:0,ER:1,ERA:5.4,WHIP:1.2,Kpct:0.0,BBpct:0.0}},
  {id:460,name:"Lukas Hlouch",team:"Czechia",pool:"Pool C",pos:"P",mlbId:693978,bats:"R",throws:"R",num:88,dob:"2000-12-12",age:26,h:193,w:98,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:0.2,SO:1,BB:0,HA:2,HR:1,ER:1,ERA:13.5,WHIP:3.0,Kpct:25.0,BBpct:0.0}},
  {id:461,name:"Marek Chlup",team:"Czechia",pool:"Pool C",pos:"RF",mlbId:803013,bats:"R",throws:"R",num:73,dob:"1999-01-09",age:27,h:193,w:100,isPitcher:false,wbcStats:{PA:14,AB:10,H:3,HR:0,R:0,RBI:0,BB:3,SO:3,SB:1,AVG:0.3,OBP:0.462}},
  {id:462,name:"Marek Krejcirik",team:"Czechia",pool:"Pool C",pos:"OF",mlbId:807163,bats:"L",throws:"R",num:97,dob:"2001-06-27",age:25,h:175,w:75,isPitcher:false,wbcStats:{PA:1,AB:1,H:0,HR:0,R:1,RBI:0,BB:0,SO:1,SB:0,AVG:0.0,OBP:0.0}},
  {id:463,name:"Marek Minarik",team:"Czechia",pool:"Pool C",pos:"P",mlbId:608042,bats:"R",throws:"R",num:15,dob:"1993-06-28",age:33,h:203,w:106,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:1.0,SO:0,BB:1,HA:2,HR:0,ER:0,ERA:0.0,WHIP:3.0,Kpct:0.0,BBpct:25.0}},
  {id:464,name:"Martin Cervenka",team:"Czechia",pool:"Pool C",pos:"C",mlbId:548431,bats:"R",throws:"R",num:55,dob:"1992-08-03",age:34,h:193,w:102,isPitcher:false,wbcStats:{PA:14,AB:14,H:2,HR:0,R:1,RBI:0,BB:0,SO:7,SB:1,AVG:0.143,OBP:0.143}},
  {id:465,name:"Martin Cervinka",team:"Czechia",pool:"Pool C",pos:"3B",mlbId:693975,bats:"R",throws:"R",num:78,dob:"1997-03-03",age:29,h:193,w:88,isPitcher:false,wbcStats:{PA:15,AB:15,H:4,HR:0,R:1,RBI:0,BB:0,SO:5,SB:0,AVG:0.267,OBP:0.267}},
  {id:466,name:"Martin Muzik",team:"Czechia",pool:"Pool C",pos:"1B",mlbId:693973,bats:"R",throws:"R",num:49,dob:"1996-04-23",age:30,h:180,w:90,isPitcher:false,wbcStats:{PA:13,AB:11,H:2,HR:0,R:0,RBI:0,BB:1,SO:5,SB:0,AVG:0.182,OBP:0.25}},
  {id:467,name:"Martin Schneider",team:"Czechia",pool:"Pool C",pos:"P",mlbId:626423,bats:"R",throws:"R",num:13,dob:"1986-03-04",age:40,h:188,w:85,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:0.1,SO:1,BB:1,HA:1,HR:0,ER:3,ERA:81.0,WHIP:6.0,Kpct:25.0,BBpct:25.0}},
  {id:468,name:"Martin Zelenka",team:"Czechia",pool:"Pool C",pos:"C",mlbId:693980,bats:"R",throws:"R",num:38,dob:"2000-12-20",age:26,h:191,w:111,isPitcher:false,wbcStats:{PA:0,AB:0,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0}},
  {id:469,name:"Matous Bubenik",team:"Czechia",pool:"Pool C",pos:"C",mlbId:828317,bats:"R",throws:"R",num:21,dob:"2005-10-28",age:21,h:191,w:102,isPitcher:false,wbcStats:{PA:4,AB:3,H:1,HR:0,R:0,RBI:0,BB:1,SO:1,SB:0,AVG:0.333,OBP:0.5}},
  {id:470,name:"Max Prejda",team:"Czechia",pool:"Pool C",pos:"OF",mlbId:838351,bats:"L",throws:"R",num:2,dob:"2007-06-06",age:19,h:183,w:79,isPitcher:false,wbcStats:{PA:9,AB:7,H:0,HR:0,R:1,RBI:0,BB:1,SO:0,SB:0,AVG:0.0,OBP:0.125}},
  {id:471,name:"Michal Kovala",team:"Czechia",pool:"Pool C",pos:"P",mlbId:807156,bats:"L",throws:"R",num:3,dob:"2003-12-28",age:23,h:183,w:83,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:1,H:0,SV:0,IP:5.0,SO:7,BB:3,HA:4,HR:2,ER:6,ERA:10.8,WHIP:1.4,Kpct:28.0,BBpct:12.0}},
  {id:472,name:"Michal Sindelka",team:"Czechia",pool:"Pool C",pos:"OF",mlbId:831799,bats:"R",throws:"R",num:11,dob:"2005-11-04",age:21,h:188,w:98,isPitcher:false,wbcStats:{PA:2,AB:2,H:0,HR:0,R:0,RBI:0,BB:0,SO:2,SB:0,AVG:0.0,OBP:0.0}},
  {id:473,name:"Milan Prokop",team:"Czechia",pool:"Pool C",pos:"IF",mlbId:807160,bats:"R",throws:"R",num:56,dob:"2003-02-12",age:23,h:183,w:88,isPitcher:false,wbcStats:{PA:9,AB:9,H:1,HR:0,R:0,RBI:0,BB:0,SO:6,SB:0,AVG:0.111,OBP:0.111}},
  {id:474,name:"Ondrej Satoria",team:"Czechia",pool:"Pool C",pos:"P",mlbId:693974,bats:"R",throws:"R",num:35,dob:"1997-02-26",age:29,h:175,w:76,isPitcher:true,wbcStats:{G:2,GS:1,W:0,L:0,H:0,SV:0,IP:8.1,SO:6,BB:1,HA:7,HR:0,ER:0,ERA:0.0,WHIP:0.96,Kpct:19.4,BBpct:3.2}},
  {id:475,name:"Ondrej Vank",team:"Czechia",pool:"Pool C",pos:"P",mlbId:838347,bats:"R",throws:"R",num:23,dob:"2005-04-25",age:21,h:185,w:87,isPitcher:true,wbcStats:{G:0,GS:0,W:0,L:0,H:0,SV:0,IP:0.0,SO:0,BB:0,HA:0,HR:0,ER:0}},
  {id:476,name:"Ryan Johnson",team:"Czechia",pool:"Pool C",pos:"1B",mlbId:689650,bats:"L",throws:"L",num:30,dob:"1992-09-29",age:34,h:196,w:102,isPitcher:false,wbcStats:{PA:2,AB:2,H:0,HR:0,R:0,RBI:0,BB:0,SO:2,SB:0,AVG:0.0,OBP:0.0}},
  {id:477,name:"Terrin Vavra",team:"Czechia",pool:"Pool C",pos:"2B",mlbId:679631,bats:"L",throws:"R",num:6,dob:"1997-05-12",age:29,h:180,w:84,isPitcher:false,wbcStats:{PA:14,AB:13,H:3,HR:1,R:1,RBI:3,BB:1,SO:0,SB:0,AVG:0.231,OBP:0.286}},
  {id:478,name:"Tomas Duffek",team:"Czechia",pool:"Pool C",pos:"P",mlbId:667766,bats:"R",throws:"L",num:7,dob:"1989-09-12",age:37,h:185,w:82,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:1.1,SO:1,BB:2,HA:1,HR:0,ER:2,ERA:13.5,WHIP:2.25,Kpct:14.3,BBpct:28.6}},
  {id:479,name:"Tomas Ondra",team:"Czechia",pool:"Pool C",pos:"P",mlbId:838345,bats:"L",throws:"R",num:57,dob:"1996-03-20",age:30,h:193,w:90,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:1,H:0,SV:0,IP:3.1,SO:2,BB:1,HA:3,HR:1,ER:3,ERA:8.1,WHIP:1.2,Kpct:14.3,BBpct:7.1}},
  {id:480,name:"Vojtech Mensik",team:"Czechia",pool:"Pool C",pos:"3B",mlbId:687781,bats:"R",throws:"R",num:77,dob:"1998-05-24",age:28,h:183,w:83,isPitcher:false,wbcStats:{PA:12,AB:10,H:1,HR:0,R:0,RBI:1,BB:1,SO:5,SB:0,AVG:0.1,OBP:0.182}},
  {id:481,name:"William Escala",team:"Czechia",pool:"Pool C",pos:"OF",mlbId:679925,bats:"S",throws:"R",num:5,dob:"1998-12-11",age:28,h:180,w:82,isPitcher:false,wbcStats:{PA:13,AB:13,H:2,HR:0,R:0,RBI:0,BB:0,SO:3,SB:0,AVG:0.154,OBP:0.154}},
  {id:482,name:"Abner Uribe",team:"Dominican Rep.",pool:"Pool D",pos:"P",mlbId:682842,bats:"R",throws:"R",num:45,dob:"2000-06-20",age:26,h:191,w:93,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:2.0,SO:5,BB:0,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.5,Kpct:71.4,BBpct:0.0}},
  {id:483,name:"Agustin Ramirez",team:"Dominican Rep.",pool:"Pool D",pos:"C",mlbId:682663,bats:"R",throws:"R",num:50,dob:"2001-09-10",age:25,h:185,w:95,isPitcher:false,wbcStats:{PA:9,AB:8,H:3,HR:0,R:1,RBI:1,BB:1,SO:1,SB:0,AVG:0.375,OBP:0.444}},
  {id:484,name:"Albert Abreu",team:"Dominican Rep.",pool:"Pool D",pos:"P",mlbId:656061,bats:"R",throws:"R",num:54,dob:"1995-09-26",age:31,h:188,w:86,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:2,SV:0,IP:4.0,SO:5,BB:3,HA:2,HR:0,ER:0,ERA:0.0,WHIP:1.25,Kpct:29.4,BBpct:17.6}},
  {id:485,name:"Amed Rosario",team:"Dominican Rep.",pool:"Pool D",pos:"2B",mlbId:642708,bats:"R",throws:"R",num:14,dob:"1995-11-20",age:31,h:185,w:86,isPitcher:false,wbcStats:{PA:6,AB:6,H:1,HR:0,R:2,RBI:0,BB:0,SO:0,SB:0,AVG:0.167,OBP:0.167}},
  {id:486,name:"Austin Wells",team:"Dominican Rep.",pool:"Pool D",pos:"C",mlbId:669224,bats:"L",throws:"R",num:28,dob:"1999-07-12",age:27,h:185,w:100,isPitcher:false,wbcStats:{PA:13,AB:13,H:2,HR:1,R:1,RBI:3,BB:0,SO:4,SB:0,AVG:0.154,OBP:0.154}},
  {id:487,name:"Brayan Bello",team:"Dominican Rep.",pool:"Pool D",pos:"P",mlbId:678394,bats:"R",throws:"R",num:66,dob:"1999-05-17",age:27,h:185,w:88,isPitcher:true,wbcStats:{G:2,GS:2,W:1,L:0,H:0,SV:0,IP:8.0,SO:9,BB:1,HA:2,HR:1,ER:1,ERA:1.12,WHIP:0.38,Kpct:32.1,BBpct:3.6}},
  {id:488,name:"Camilo Doval",team:"Dominican Rep.",pool:"Pool D",pos:"P",mlbId:666808,bats:"R",throws:"R",num:75,dob:"1997-07-04",age:29,h:188,w:93,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:2.0,SO:1,BB:0,HA:2,HR:0,ER:0,ERA:0.0,WHIP:1.0,Kpct:12.5,BBpct:0.0}},
  {id:489,name:"Carlos Estevez",team:"Dominican Rep.",pool:"Pool D",pos:"P",mlbId:608032,bats:"R",throws:"R",num:53,dob:"1992-12-28",age:34,h:198,w:126,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:2.0,SO:1,BB:1,HA:2,HR:0,ER:0,ERA:0.0,WHIP:1.5,Kpct:12.5,BBpct:12.5}},
  {id:490,name:"Carlos Santana",team:"Dominican Rep.",pool:"Pool D",pos:"1B",mlbId:467793,bats:"S",throws:"R",num:41,dob:"1986-04-08",age:40,h:178,w:95,isPitcher:false,wbcStats:{PA:10,AB:9,H:4,HR:0,R:1,RBI:2,BB:1,SO:1,SB:0,AVG:0.444,OBP:0.5}},
  {id:491,name:"Cristopher Sanchez",team:"Dominican Rep.",pool:"Pool D",pos:"P",mlbId:650911,bats:"L",throws:"L",num:61,dob:"1996-12-12",age:30,h:198,w:91,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:1.1,SO:4,BB:1,HA:6,HR:0,ER:3,ERA:20.25,WHIP:5.25,Kpct:33.3,BBpct:8.3}},
  {id:492,name:"Dennis Santana",team:"Dominican Rep.",pool:"Pool D",pos:"P",mlbId:642701,bats:"R",throws:"R",num:60,dob:"1996-04-12",age:30,h:188,w:92,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.2,SO:1,BB:1,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.6,Kpct:16.7,BBpct:16.7}},
  {id:493,name:"Elvis Alvarado",team:"Dominican Rep.",pool:"Pool D",pos:"P",mlbId:665660,bats:"R",throws:"R",num:37,dob:"1999-02-23",age:27,h:193,w:83,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:1,SV:0,IP:1.2,SO:2,BB:0,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.6,Kpct:33.3,BBpct:0.0}},
  {id:494,name:"Erik Gonzalez",team:"Dominican Rep.",pool:"Pool D",pos:"SS",mlbId:570481,bats:"R",throws:"R",num:11,dob:"1991-08-31",age:35,h:188,w:93,isPitcher:false,wbcStats:{PA:5,AB:5,H:2,HR:0,R:2,RBI:1,BB:0,SO:1,SB:0,AVG:0.4,OBP:0.4}},
  {id:495,name:"Fernando Tatis Jr.",team:"Dominican Rep.",pool:"Pool D",pos:"RF",mlbId:665487,bats:"R",throws:"R",num:23,dob:"1999-01-02",age:27,h:191,w:98,isPitcher:false,wbcStats:{PA:19,AB:13,H:8,HR:1,R:7,RBI:8,BB:6,SO:2,SB:0,AVG:0.615,OBP:0.737}},
  {id:496,name:"Geraldo Perdomo",team:"Dominican Rep.",pool:"Pool D",pos:"SS",mlbId:672695,bats:"S",throws:"R",num:2,dob:"1999-10-22",age:27,h:188,w:92,isPitcher:false,wbcStats:{PA:13,AB:9,H:2,HR:0,R:5,RBI:2,BB:4,SO:0,SB:2,AVG:0.222,OBP:0.462}},
  {id:497,name:"Gregory Soto",team:"Dominican Rep.",pool:"Pool D",pos:"P",mlbId:642397,bats:"L",throws:"L",num:65,dob:"1995-02-11",age:31,h:185,w:112,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:2.1,SO:6,BB:2,HA:3,HR:0,ER:0,ERA:0.0,WHIP:2.14,Kpct:50.0,BBpct:16.7}},
  {id:498,name:"Huascar Brazoban",team:"Dominican Rep.",pool:"Pool D",pos:"P",mlbId:623211,bats:"R",throws:"R",num:43,dob:"1989-10-15",age:37,h:191,w:70,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:2.1,SO:3,BB:0,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.0,Kpct:42.9,BBpct:0.0}},
  {id:499,name:"Juan Mejia",team:"Dominican Rep.",pool:"Pool D",pos:"P",mlbId:675848,bats:"R",throws:"R",num:47,dob:"2000-07-04",age:26,h:191,w:91,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.1,SO:0,BB:0,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.0,Kpct:0.0,BBpct:0.0}},
  {id:500,name:"Juan Soto",team:"Dominican Rep.",pool:"Pool D",pos:"RF",mlbId:665742,bats:"L",throws:"L",num:22,dob:"1998-10-25",age:28,h:185,w:102,isPitcher:false,wbcStats:{PA:22,AB:18,H:7,HR:3,R:5,RBI:8,BB:4,SO:4,SB:1,AVG:0.389,OBP:0.5}},
  {id:501,name:"Julio Rodriguez",team:"Dominican Rep.",pool:"Pool D",pos:"CF",mlbId:677594,bats:"R",throws:"R",num:44,dob:"2000-12-29",age:26,h:193,w:103,isPitcher:false,wbcStats:{PA:14,AB:9,H:3,HR:1,R:3,RBI:3,BB:5,SO:2,SB:0,AVG:0.333,OBP:0.571}},
  {id:502,name:"Junior Caminero",team:"Dominican Rep.",pool:"Pool D",pos:"3B",mlbId:691406,bats:"R",throws:"R",num:13,dob:"2003-07-05",age:23,h:185,w:100,isPitcher:false,wbcStats:{PA:18,AB:16,H:10,HR:3,R:4,RBI:6,BB:1,SO:0,SB:0,AVG:0.625,OBP:0.647}},
  {id:503,name:"Junior Lake",team:"Dominican Rep.",pool:"Pool D",pos:"LF",mlbId:516809,bats:"R",throws:"R",num:0,dob:"1990-03-27",age:36,h:188,w:104,isPitcher:false,wbcStats:{PA:5,AB:5,H:1,HR:0,R:1,RBI:0,BB:0,SO:1,SB:0,AVG:0.2,OBP:0.2}},
  {id:504,name:"Ketel Marte",team:"Dominican Rep.",pool:"Pool D",pos:"2B",mlbId:606466,bats:"S",throws:"R",num:4,dob:"1993-10-12",age:33,h:183,w:95,isPitcher:false,wbcStats:{PA:18,AB:11,H:2,HR:0,R:5,RBI:3,BB:4,SO:1,SB:0,AVG:0.182,OBP:0.4}},
  {id:505,name:"Luis Severino",team:"Dominican Rep.",pool:"Pool D",pos:"P",mlbId:622663,bats:"R",throws:"R",num:40,dob:"1994-02-20",age:32,h:188,w:99,isPitcher:true,wbcStats:{G:2,GS:2,W:1,L:0,H:0,SV:0,IP:6.0,SO:7,BB:0,HA:8,HR:2,ER:3,ERA:4.5,WHIP:1.33,Kpct:25.9,BBpct:0.0}},
  {id:506,name:"Manny Machado",team:"Dominican Rep.",pool:"Pool D",pos:"3B",mlbId:592518,bats:"R",throws:"R",num:3,dob:"1992-07-06",age:34,h:188,w:99,isPitcher:false,wbcStats:{PA:20,AB:12,H:4,HR:1,R:6,RBI:1,BB:7,SO:1,SB:0,AVG:0.333,OBP:0.579}},
  {id:507,name:"Oneil Cruz",team:"Dominican Rep.",pool:"Pool D",pos:"CF",mlbId:665833,bats:"L",throws:"R",num:15,dob:"1998-10-04",age:28,h:201,w:112,isPitcher:false,wbcStats:{PA:9,AB:7,H:3,HR:2,R:4,RBI:4,BB:2,SO:0,SB:0,AVG:0.429,OBP:0.556}},
  {id:508,name:"Sandy Alcantara",team:"Dominican Rep.",pool:"Pool D",pos:"P",mlbId:645261,bats:"R",throws:"R",num:7,dob:"1995-09-07",age:31,h:196,w:91,isPitcher:true,wbcStats:{G:0,GS:0,W:0,L:0,H:0,SV:0,IP:0.0,SO:0,BB:0,HA:0,HR:0,ER:0}},
  {id:509,name:"Seranthony Dominguez",team:"Dominican Rep.",pool:"Pool D",pos:"P",mlbId:622554,bats:"R",throws:"R",num:48,dob:"1994-11-25",age:32,h:185,w:102,isPitcher:true,wbcStats:{G:3,GS:0,W:2,L:0,H:0,SV:0,IP:2.1,SO:3,BB:0,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.43,Kpct:37.5,BBpct:0.0}},
  {id:510,name:"Vladimir Guerrero Jr.",team:"Dominican Rep.",pool:"Pool D",pos:"1B",mlbId:665489,bats:"R",throws:"R",num:27,dob:"1999-03-16",age:27,h:183,w:111,isPitcher:false,wbcStats:{PA:15,AB:13,H:4,HR:1,R:2,RBI:6,BB:1,SO:1,SB:0,AVG:0.308,OBP:0.357}},
  {id:511,name:"Wandy Peralta",team:"Dominican Rep.",pool:"Pool D",pos:"P",mlbId:593974,bats:"L",throws:"L",num:58,dob:"1991-07-27",age:35,h:183,w:103,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:3.0,SO:1,BB:0,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.0,Kpct:10.0,BBpct:0.0}},
  {id:512,name:"Been Gwak",team:"Korea",pool:"Pool C",pos:"P",mlbId:808971,bats:"R",throws:"R",num:47,dob:"1999-05-28",age:27,h:188,w:95,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:3.1,SO:3,BB:1,HA:2,HR:1,ER:1,ERA:2.7,WHIP:0.9,Kpct:23.1,BBpct:7.7}},
  {id:513,name:"Bo Gyeong Moon",team:"Korea",pool:"Pool C",pos:"3B",mlbId:823575,bats:"L",throws:"R",num:2,dob:"2000-07-19",age:26,h:183,w:88,isPitcher:false,wbcStats:{PA:16,AB:13,H:7,HR:2,R:3,RBI:11,BB:2,SO:1,SB:0,AVG:0.538,OBP:0.6}},
  {id:514,name:"Byeong Hyeon Jo",team:"Korea",pool:"Pool C",pos:"P",mlbId:823568,bats:"R",throws:"R",num:11,dob:"2002-05-08",age:24,h:183,w:90,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:0,SV:0,IP:4.0,SO:4,BB:3,HA:1,HR:1,ER:1,ERA:2.25,WHIP:1.0,Kpct:26.7,BBpct:20.0}},
  {id:515,name:"Dane Dunning",team:"Korea",pool:"Pool C",pos:"P",mlbId:641540,bats:"R",throws:"R",num:33,dob:"1994-12-20",age:32,h:193,w:102,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:2.2,SO:2,BB:1,HA:3,HR:1,ER:2,ERA:6.75,WHIP:1.5,Kpct:20.0,BBpct:10.0}},
  {id:516,name:"Do Yeong Kim",team:"Korea",pool:"Pool C",pos:"3B",mlbId:838339,bats:"R",throws:"R",num:5,dob:"2003-10-02",age:23,h:183,w:85,isPitcher:false,wbcStats:{PA:19,AB:17,H:4,HR:1,R:3,RBI:4,BB:2,SO:2,SB:0,AVG:0.235,OBP:0.316}},
  {id:517,name:"Dong Won Park",team:"Korea",pool:"Pool C",pos:"C",mlbId:673526,bats:"R",throws:"R",num:27,dob:"1990-04-07",age:36,h:178,w:92,isPitcher:false,wbcStats:{PA:13,AB:11,H:2,HR:0,R:3,RBI:0,BB:2,SO:7,SB:0,AVG:0.182,OBP:0.308}},
  {id:518,name:"Hae Min Park",team:"Korea",pool:"Pool C",pos:"CF",mlbId:673527,bats:"L",throws:"R",num:17,dob:"1990-02-24",age:36,h:180,w:75,isPitcher:false,wbcStats:{PA:2,AB:1,H:0,HR:0,R:2,RBI:0,BB:0,SO:1,SB:0,AVG:0.0,OBP:0.0}},
  {id:519,name:"Hyeong Jun So",team:"Korea",pool:"Pool C",pos:"P",mlbId:808984,bats:"R",throws:"R",num:30,dob:"2001-09-16",age:25,h:188,w:92,isPitcher:true,wbcStats:{G:2,GS:1,W:1,L:0,H:0,SV:0,IP:5.0,SO:4,BB:1,HA:5,HR:1,ER:1,ERA:1.8,WHIP:1.2,Kpct:21.1,BBpct:5.3}},
  {id:520,name:"Hyeseong Kim",team:"Korea",pool:"Pool C",pos:"2B",mlbId:808975,bats:"L",throws:"R",num:3,dob:"1999-01-27",age:27,h:178,w:79,isPitcher:false,wbcStats:{PA:12,AB:10,H:1,HR:1,R:2,RBI:3,BB:2,SO:4,SB:1,AVG:0.1,OBP:0.25}},
  {id:521,name:"Hyun Bin Moon",team:"Korea",pool:"Pool C",pos:"LF",mlbId:806806,bats:"L",throws:"R",num:51,dob:"2004-04-20",age:22,h:175,w:82,isPitcher:false,wbcStats:{PA:3,AB:2,H:0,HR:0,R:0,RBI:0,BB:1,SO:1,SB:0,AVG:0.0,OBP:0.333}},
  {id:522,name:"Hyun Jin Ryu",team:"Korea",pool:"Pool C",pos:"P",mlbId:547943,bats:"R",throws:"L",num:99,dob:"1987-03-25",age:39,h:191,w:113,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:3.0,SO:3,BB:0,HA:3,HR:1,ER:1,ERA:3.0,WHIP:1.0,Kpct:25.0,BBpct:0.0}},
  {id:523,name:"Hyun Min Ahn",team:"Korea",pool:"Pool C",pos:"OF",mlbId:838336,bats:"R",throws:"R",num:23,dob:"2003-08-22",age:23,h:183,w:90,isPitcher:false,wbcStats:{PA:16,AB:12,H:4,HR:0,R:4,RBI:1,BB:3,SO:6,SB:1,AVG:0.333,OBP:0.467}},
  {id:524,name:"Hyung Jun Kim",team:"Korea",pool:"Pool C",pos:"C",mlbId:823570,bats:"R",throws:"R",num:25,dob:"1999-11-02",age:27,h:188,w:98,isPitcher:false,wbcStats:{PA:1,AB:0,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0}},
  {id:525,name:"Ja Wook Koo",team:"Korea",pool:"Pool C",pos:"LF",mlbId:838332,bats:"L",throws:"R",num:65,dob:"1993-02-12",age:33,h:188,w:75,isPitcher:false,wbcStats:{PA:2,AB:2,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0,AVG:0.0,OBP:0.0}},
  {id:526,name:"Jahmai Jones",team:"Korea",pool:"Pool C",pos:"LF",mlbId:663330,bats:"R",throws:"R",num:15,dob:"1997-08-04",age:29,h:178,w:95,isPitcher:false,wbcStats:{PA:19,AB:18,H:4,HR:1,R:3,RBI:2,BB:1,SO:3,SB:1,AVG:0.222,OBP:0.263}},
  {id:527,name:"Ju Won Kim",team:"Korea",pool:"Pool C",pos:"SS",mlbId:823572,bats:"R",throws:"R",num:7,dob:"2002-07-30",age:24,h:185,w:83,isPitcher:false,wbcStats:{PA:15,AB:14,H:3,HR:0,R:1,RBI:1,BB:0,SO:6,SB:0,AVG:0.214,OBP:0.214}},
  {id:528,name:"Ju Young Son",team:"Korea",pool:"Pool C",pos:"P",mlbId:838333,bats:"L",throws:"L",num:29,dob:"1998-12-02",age:28,h:191,w:95,isPitcher:true,wbcStats:{G:2,GS:1,W:1,L:0,H:0,SV:0,IP:2.0,SO:1,BB:1,HA:2,HR:0,ER:0,ERA:0.0,WHIP:1.5,Kpct:11.1,BBpct:11.1}},
  {id:529,name:"Jung Hoo Lee",team:"Korea",pool:"Pool C",pos:"CF",mlbId:808982,bats:"L",throws:"R",num:22,dob:"1998-08-20",age:28,h:183,w:89,isPitcher:false,wbcStats:{PA:19,AB:18,H:5,HR:0,R:4,RBI:2,BB:1,SO:0,SB:0,AVG:0.278,OBP:0.316}},
  {id:530,name:"Kyung Eun Noh",team:"Korea",pool:"Pool C",pos:"P",mlbId:628359,bats:"R",throws:"R",num:38,dob:"1984-03-11",age:42,h:188,w:84,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:0,H:1,SV:0,IP:3.1,SO:2,BB:1,HA:3,HR:0,ER:0,ERA:0.0,WHIP:1.2,Kpct:15.4,BBpct:7.7}},
  {id:531,name:"Min Jae Shin",team:"Korea",pool:"Pool C",pos:"2B",mlbId:823593,bats:"L",throws:"R",num:4,dob:"1996-01-21",age:30,h:170,w:67,isPitcher:false,wbcStats:{PA:4,AB:4,H:0,HR:0,R:1,RBI:0,BB:0,SO:0,SB:0,AVG:0.0,OBP:0.0}},
  {id:532,name:"Riley O'Brien",team:"Korea",pool:"Pool C",pos:"P",mlbId:676617,bats:"R",throws:"R",num:55,dob:"1995-02-06",age:31,h:193,w:82,isPitcher:true,wbcStats:{G:0,GS:0,W:0,L:0,H:0,SV:0,IP:0.0,SO:0,BB:0,HA:0,HR:0,ER:0}},
  {id:533,name:"Seung Ki Song",team:"Korea",pool:"Pool C",pos:"P",mlbId:838334,bats:"L",throws:"L",num:66,dob:"2002-04-10",age:24,h:180,w:90,isPitcher:true,wbcStats:{G:0,GS:0,W:0,L:0,H:0,SV:0,IP:0.0,SO:0,BB:0,HA:0,HR:0,ER:0}},
  {id:534,name:"Shay Whitcomb",team:"Korea",pool:"Pool C",pos:"2B",mlbId:694376,bats:"R",throws:"R",num:10,dob:"1998-09-28",age:28,h:183,w:92,isPitcher:false,wbcStats:{PA:13,AB:12,H:3,HR:2,R:2,RBI:3,BB:1,SO:3,SB:0,AVG:0.25,OBP:0.308}},
  {id:535,name:"Si Hwan Roh",team:"Korea",pool:"Pool C",pos:"3B",mlbId:823574,bats:"R",throws:"R",num:8,dob:"2000-12-03",age:26,h:185,w:105,isPitcher:false,wbcStats:{PA:3,AB:2,H:0,HR:0,R:0,RBI:0,BB:1,SO:2,SB:1,AVG:0.0,OBP:0.333}},
  {id:536,name:"Taek Yeon Kim",team:"Korea",pool:"Pool C",pos:"P",mlbId:823559,bats:"R",throws:"R",num:63,dob:"2005-06-03",age:21,h:180,w:88,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.2,SO:2,BB:1,HA:2,HR:0,ER:1,ERA:5.4,WHIP:1.8,Kpct:25.0,BBpct:12.5}},
  {id:537,name:"Woo Joo Jeong",team:"Korea",pool:"Pool C",pos:"P",mlbId:838331,bats:"R",throws:"R",num:61,dob:"2006-11-07",age:20,h:183,w:88,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:1.0,SO:2,BB:0,HA:2,HR:1,ER:3,ERA:27.0,WHIP:2.0,Kpct:33.3,BBpct:0.0}},
  {id:538,name:"Woo Suk Go",team:"Korea",pool:"Pool C",pos:"P",mlbId:808970,bats:"R",throws:"R",num:19,dob:"1998-08-06",age:28,h:180,w:90,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:1,H:0,SV:0,IP:2.2,SO:1,BB:0,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.0,Kpct:11.1,BBpct:0.0}},
  {id:539,name:"Yeong Chan You",team:"Korea",pool:"Pool C",pos:"P",mlbId:823590,bats:"R",throws:"R",num:54,dob:"1997-03-07",age:29,h:185,w:90,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:1.0,SO:2,BB:1,HA:1,HR:0,ER:1,ERA:9.0,WHIP:2.0,Kpct:40.0,BBpct:20.0}},
  {id:540,name:"Yeong Hyun Park",team:"Korea",pool:"Pool C",pos:"P",mlbId:823561,bats:"R",throws:"R",num:60,dob:"2003-10-11",age:23,h:183,w:91,isPitcher:true,wbcStats:{G:3,GS:0,W:0,L:1,H:0,SV:0,IP:2.2,SO:3,BB:3,HA:0,HR:0,ER:2,ERA:6.75,WHIP:1.12,Kpct:27.3,BBpct:27.3}},
  {id:541,name:"Young Kyu Kim",team:"Korea",pool:"Pool C",pos:"P",mlbId:838342,bats:"L",throws:"L",num:14,dob:"2000-02-10",age:26,h:188,w:86,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:1.0,SO:2,BB:2,HA:1,HR:0,ER:1,ERA:9.0,WHIP:3.0,Kpct:33.3,BBpct:33.3}},
  {id:542,name:"Young Pyo Ko",team:"Korea",pool:"Pool C",pos:"P",mlbId:808978,bats:"R",throws:"R",num:1,dob:"1991-09-16",age:35,h:185,w:88,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:2.2,SO:4,BB:1,HA:3,HR:3,ER:4,ERA:13.5,WHIP:1.5,Kpct:33.3,BBpct:8.3}},
  {id:543,name:"Atsuki Taneichi",team:"Japan",pool:"Pool C",pos:"P",mlbId:684001,bats:"R",throws:"R",num:26,dob:"1998-09-07",age:28,h:183,w:88,isPitcher:true,wbcStats:{G:2,GS:0,W:1,L:0,H:1,SV:0,IP:2.0,SO:5,BB:0,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.0,Kpct:83.3,BBpct:0.0}},
  {id:544,name:"Chihiro Sumida",team:"Japan",pool:"Pool C",pos:"P",mlbId:838757,bats:"L",throws:"L",num:22,dob:"1999-08-20",age:27,h:175,w:76,isPitcher:true,wbcStats:{G:1,GS:0,W:1,L:0,H:0,SV:0,IP:3.0,SO:7,BB:0,HA:2,HR:0,ER:0,ERA:0.0,WHIP:0.67,Kpct:63.6,BBpct:0.0}},
  {id:545,name:"Hiromi Itoh",team:"Japan",pool:"Pool C",pos:"P",mlbId:808955,bats:"L",throws:"R",num:14,dob:"1997-08-31",age:29,h:178,w:82,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:3.0,SO:6,BB:0,HA:1,HR:1,ER:2,ERA:6.0,WHIP:0.33,Kpct:54.5,BBpct:0.0}},
  {id:546,name:"Hiroto Takahashi",team:"Japan",pool:"Pool C",pos:"P",mlbId:808964,bats:"R",throws:"R",num:28,dob:"2002-08-09",age:24,h:188,w:86,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:4.2,SO:5,BB:1,HA:2,HR:0,ER:0,ERA:0.0,WHIP:0.64,Kpct:31.2,BBpct:6.2}},
  {id:547,name:"Hiroya Miyagi",team:"Japan",pool:"Pool C",pos:"P",mlbId:808958,bats:"L",throws:"L",num:13,dob:"2001-08-25",age:25,h:170,w:78,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:3.1,SO:4,BB:2,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.6,Kpct:30.8,BBpct:15.4}},
  {id:548,name:"Kaito Kozono",team:"Japan",pool:"Pool C",pos:"SS",mlbId:838344,bats:"L",throws:"R",num:3,dob:"2000-06-07",age:26,h:178,w:86,isPitcher:false,wbcStats:{PA:4,AB:3,H:1,HR:0,R:1,RBI:0,BB:1,SO:0,SB:0,AVG:0.333,OBP:0.5}},
  {id:549,name:"Kazuma Okamoto",team:"Japan",pool:"Pool C",pos:"3B",mlbId:672960,bats:"R",throws:"R",num:25,dob:"1996-06-30",age:30,h:185,w:100,isPitcher:false,wbcStats:{PA:18,AB:15,H:2,HR:0,R:1,RBI:1,BB:3,SO:3,SB:0,AVG:0.133,OBP:0.278}},
  {id:550,name:"Kensuke Kondoh",team:"Japan",pool:"Pool C",pos:"OF",mlbId:685547,bats:"L",throws:"R",num:8,dob:"1993-08-09",age:33,h:170,w:86,isPitcher:false,wbcStats:{PA:13,AB:12,H:0,HR:0,R:2,RBI:0,BB:1,SO:2,SB:0,AVG:0.0,OBP:0.077}},
  {id:551,name:"Kenya Wakatsuki",team:"Japan",pool:"Pool C",pos:"C",mlbId:838350,bats:"R",throws:"R",num:4,dob:"1995-10-04",age:31,h:180,w:88,isPitcher:false,wbcStats:{PA:9,AB:7,H:3,HR:0,R:1,RBI:1,BB:2,SO:0,SB:0,AVG:0.429,OBP:0.556}},
  {id:552,name:"Koki Kitayama",team:"Japan",pool:"Pool C",pos:"P",mlbId:838348,bats:"R",throws:"R",num:57,dob:"1999-04-10",age:27,h:183,w:86,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:0,IP:2.0,SO:4,BB:0,HA:1,HR:0,ER:0,ERA:0.0,WHIP:0.5,Kpct:57.1,BBpct:0.0}},
  {id:553,name:"Masataka Yoshida",team:"Japan",pool:"Pool C",pos:"LF",mlbId:807799,bats:"L",throws:"R",num:34,dob:"1993-07-15",age:33,h:173,w:87,isPitcher:false,wbcStats:{PA:14,AB:12,H:6,HR:2,R:4,RBI:6,BB:2,SO:1,SB:0,AVG:0.5,OBP:0.571}},
  {id:554,name:"Munetaka Murakami",team:"Japan",pool:"Pool C",pos:"3B",mlbId:808959,bats:"L",throws:"R",num:55,dob:"2000-02-02",age:26,h:188,w:97,isPitcher:false,wbcStats:{PA:17,AB:15,H:3,HR:1,R:4,RBI:5,BB:2,SO:4,SB:1,AVG:0.2,OBP:0.294}},
  {id:555,name:"Ryuhei Sotani",team:"Japan",pool:"Pool C",pos:"P",mlbId:838349,bats:"L",throws:"L",num:47,dob:"2000-11-30",age:26,h:183,w:83,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:1.0,SO:1,BB:0,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.0,Kpct:33.3,BBpct:0.0}},
  {id:556,name:"Seishiro Sakamoto",team:"Japan",pool:"Pool C",pos:"C",mlbId:831656,bats:"R",throws:"R",num:12,dob:"1993-11-10",age:33,h:178,w:79,isPitcher:false,wbcStats:{PA:2,AB:2,H:0,HR:0,R:0,RBI:0,BB:0,SO:2,SB:0,AVG:0.0,OBP:0.0}},
  {id:557,name:"Seiya Suzuki",team:"Japan",pool:"Pool C",pos:"RF",mlbId:673548,bats:"R",throws:"R",num:51,dob:"1994-08-18",age:32,h:180,w:83,isPitcher:false,wbcStats:{PA:14,AB:9,H:3,HR:2,R:4,RBI:5,BB:5,SO:1,SB:0,AVG:0.333,OBP:0.571}},
  {id:558,name:"Shohei Ohtani",team:"Japan",pool:"Pool C",pos:"DH",mlbId:660271,bats:"L",throws:"R",num:16,dob:"1994-07-05",age:32,h:193,w:95,isPitcher:false,wbcStats:{PA:13,AB:9,H:5,HR:2,R:4,RBI:6,BB:4,SO:0,SB:0,AVG:0.556,OBP:0.692}},
  {id:559,name:"Shoma Fujihira",team:"Japan",pool:"Pool C",pos:"P",mlbId:838755,bats:"R",throws:"R",num:46,dob:"1998-09-21",age:28,h:185,w:85,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:0,SV:0,IP:0.1,SO:1,BB:0,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.0,Kpct:100.0,BBpct:0.0}},
  {id:560,name:"Shota Morishita",team:"Japan",pool:"Pool C",pos:"RF",mlbId:831674,bats:"R",throws:"R",num:23,dob:"2000-08-14",age:26,h:183,w:93,isPitcher:false,wbcStats:{PA:7,AB:6,H:1,HR:0,R:1,RBI:0,BB:1,SO:0,SB:0,AVG:0.167,OBP:0.286}},
  {id:561,name:"Shugo Maki",team:"Japan",pool:"Pool C",pos:"2B",mlbId:808957,bats:"R",throws:"R",num:2,dob:"1998-04-21",age:28,h:178,w:93,isPitcher:false,wbcStats:{PA:12,AB:9,H:2,HR:0,R:3,RBI:1,BB:3,SO:2,SB:0,AVG:0.222,OBP:0.417}},
  {id:562,name:"Sosuke Genda",team:"Japan",pool:"Pool C",pos:"SS",mlbId:683821,bats:"L",throws:"R",num:6,dob:"1993-02-16",age:33,h:180,w:75,isPitcher:false,wbcStats:{PA:12,AB:7,H:4,HR:0,R:3,RBI:4,BB:3,SO:0,SB:0,AVG:0.571,OBP:0.7}},
  {id:563,name:"Taisei Makihara",team:"Japan",pool:"Pool C",pos:"SS",mlbId:611463,bats:"L",throws:"R",num:5,dob:"1992-10-15",age:34,h:173,w:68,isPitcher:false,wbcStats:{PA:6,AB:4,H:1,HR:0,R:2,RBI:0,BB:1,SO:3,SB:0,AVG:0.25,OBP:0.4}},
  {id:564,name:"Taisei Ota",team:"Japan",pool:"Pool C",pos:"P",mlbId:808962,bats:"R",throws:"R",num:15,dob:"1999-06-29",age:27,h:180,w:88,isPitcher:true,wbcStats:{G:2,GS:0,W:0,L:0,H:0,SV:2,IP:2.0,SO:1,BB:0,HA:2,HR:2,ER:2,ERA:9.0,WHIP:1.0,Kpct:12.5,BBpct:0.0}},
  {id:565,name:"Teruaki Sato",team:"Japan",pool:"Pool C",pos:"3B",mlbId:831664,bats:"L",throws:"R",num:7,dob:"1999-03-13",age:27,h:188,w:95,isPitcher:false,wbcStats:{PA:8,AB:6,H:2,HR:0,R:1,RBI:1,BB:1,SO:0,SB:0,AVG:0.333,OBP:0.429}},
  {id:566,name:"Tomoyuki Sugano",team:"Japan",pool:"Pool C",pos:"P",mlbId:608372,bats:"R",throws:"R",num:19,dob:"1989-10-11",age:37,h:185,w:104,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:4.0,SO:2,BB:0,HA:4,HR:0,ER:0,ERA:0.0,WHIP:1.0,Kpct:13.3,BBpct:0.0}},
  {id:567,name:"Ukyo Shuto",team:"Japan",pool:"Pool C",pos:"CF",mlbId:683970,bats:"L",throws:"R",num:20,dob:"1996-02-10",age:30,h:180,w:70,isPitcher:false,wbcStats:{PA:4,AB:4,H:2,HR:1,R:2,RBI:3,BB:0,SO:1,SB:3,AVG:0.5,OBP:0.5}},
  {id:568,name:"Yoshinobu Yamamoto",team:"Japan",pool:"Pool C",pos:"P",mlbId:808967,bats:"R",throws:"R",num:18,dob:"1998-08-17",age:28,h:178,w:80,isPitcher:true,wbcStats:{G:1,GS:1,W:1,L:0,H:0,SV:0,IP:2.2,SO:2,BB:3,HA:0,HR:0,ER:0,ERA:0.0,WHIP:1.12,Kpct:18.2,BBpct:27.3}},
  {id:569,name:"Yuhei Nakamura",team:"Japan",pool:"Pool C",pos:"C",mlbId:673520,bats:"R",throws:"R",num:27,dob:"1990-06-17",age:36,h:175,w:79,isPitcher:false,wbcStats:{PA:4,AB:3,H:2,HR:0,R:1,RBI:0,BB:0,SO:0,SB:0,AVG:0.667,OBP:0.667}},
  {id:570,name:"Yuki Matsumoto",team:"Japan",pool:"Pool C",pos:"P",mlbId:838346,bats:"L",throws:"R",num:66,dob:"1996-04-14",age:30,h:183,w:92,isPitcher:true,wbcStats:{G:1,GS:0,W:0,L:0,H:1,SV:0,IP:1.0,SO:2,BB:2,HA:2,HR:0,ER:1,ERA:9.0,WHIP:4.0,Kpct:28.6,BBpct:28.6}},
  {id:571,name:"Yumeto Kanemaru",team:"Japan",pool:"Pool C",pos:"P",mlbId:838756,bats:"L",throws:"L",num:24,dob:"2003-02-01",age:23,h:178,w:77,isPitcher:true,wbcStats:{G:1,GS:0,W:1,L:0,H:0,SV:0,IP:2.0,SO:5,BB:0,HA:0,HR:0,ER:0,ERA:0.0,WHIP:0.0,Kpct:83.3,BBpct:0.0}},
  {id:572,name:"Yusei Kikuchi",team:"Japan",pool:"Pool C",pos:"P",mlbId:579328,bats:"L",throws:"L",num:17,dob:"1991-06-17",age:35,h:183,w:95,isPitcher:true,wbcStats:{G:1,GS:1,W:0,L:0,H:0,SV:0,IP:3.0,SO:4,BB:0,HA:6,HR:0,ER:3,ERA:9.0,WHIP:2.0,Kpct:26.7,BBpct:0.0}},
  {id:573,name:"Abdiel Mendoza",team:"Panama",pool:"Pool A",pos:"SS",mlbId:665959,bats:"R",throws:"R",num:1,dob:"2002-05-19",age:23,h:185,w:82,isPitcher:false,wbcStats:{PA:0,AB:0,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0}},
  {id:574,name:"Leo Bernal",team:"Panama",pool:"Pool A",pos:"P",mlbId:699024,bats:"R",throws:"R",num:17,dob:"2001-10-02",age:24,h:190,w:93,isPitcher:true,wbcStats:{G:0,GS:0,W:0,L:0,H:0,SV:0,IP:0}},
  {id:575,name:"Christian Bethancourt",team:"Panama",pool:"Pool A",pos:"C",mlbId:542194,bats:"R",throws:"R",num:14,dob:"1991-09-02",age:34,h:191,w:104,isPitcher:false,wbcStats:{PA:0,AB:0,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0}},
  {id:576,name:"Enrique Bradfield Jr.",team:"Panama",pool:"Pool A",pos:"OF",mlbId:690961,bats:"L",throws:"L",num:5,dob:"2002-06-17",age:23,h:183,w:79,isPitcher:false,wbcStats:{PA:0,AB:0,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0}},
  {id:577,name:"José Caballero",team:"Panama",pool:"Pool A",pos:"IF",mlbId:676609,bats:"R",throws:"R",num:7,dob:"1998-07-27",age:27,h:183,w:86,isPitcher:false,wbcStats:{PA:0,AB:0,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0}},
  {id:578,name:"Miguel Cienfuegos",team:"Panama",pool:"Pool A",pos:"P",mlbId:800550,bats:"R",throws:"R",num:21,dob:"2003-01-15",age:23,h:196,w:107,isPitcher:true,wbcStats:{G:0,GS:0,W:0,L:0,H:0,SV:0,IP:0}},
  {id:579,name:"James Gonzalez",team:"Panama",pool:"Pool A",pos:"P",mlbId:686857,bats:"R",throws:"R",num:28,dob:"2001-08-23",age:24,h:196,w:113,isPitcher:true,wbcStats:{G:0,GS:0,W:0,L:0,H:0,SV:0,IP:0}},
  {id:580,name:"Javy Guerra",team:"Panama",pool:"Pool A",pos:"P",mlbId:642770,bats:"R",throws:"R",num:62,dob:"1995-10-28",age:30,h:191,w:106,isPitcher:true,wbcStats:{G:0,GS:0,W:0,L:0,H:0,SV:0,IP:0}},
  {id:581,name:"Iván Herrera",team:"Panama",pool:"Pool A",pos:"C",mlbId:671056,bats:"R",throws:"R",num:48,dob:"2000-06-01",age:25,h:178,w:95,isPitcher:false,wbcStats:{PA:0,AB:0,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0}},
  {id:582,name:"Leo Jiménez",team:"Panama",pool:"Pool A",pos:"SS",mlbId:677870,bats:"S",throws:"R",num:9,dob:"2001-05-17",age:24,h:178,w:84,isPitcher:false,wbcStats:{PA:0,AB:0,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0}},
  {id:583,name:"Miguel Amaya",team:"Panama",pool:"Pool A",pos:"C",mlbId:665804,bats:"R",throws:"R",num:59,dob:"2000-03-09",age:26,h:185,w:108,isPitcher:false,wbcStats:{PA:0,AB:0,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0}},
  {id:584,name:"Jose Ramos",team:"Panama",pool:"Pool A",pos:"OF",mlbId:682947,bats:"R",throws:"R",num:15,dob:"2002-07-29",age:23,h:193,w:104,isPitcher:false,wbcStats:{PA:0,AB:0,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0}},
  {id:585,name:"Erian Rodriguez",team:"Panama",pool:"Pool A",pos:"P",mlbId:701477,bats:"R",throws:"R",num:30,dob:"2002-04-24",age:23,h:188,w:98,isPitcher:true,wbcStats:{G:0,GS:0,W:0,L:0,H:0,SV:0,IP:0}},
  {id:586,name:"Edmundo Sosa",team:"Panama",pool:"Pool A",pos:"IF",mlbId:624641,bats:"R",throws:"R",num:5,dob:"1996-03-06",age:30,h:178,w:79,isPitcher:false,wbcStats:{PA:0,AB:0,H:0,HR:0,R:0,RBI:0,BB:0,SO:0,SB:0}},
];

// ── MLB StatsAPI 훅 — 2025 + 2024 각각 fetch ────────────────────────────────
// 선수 사진 URL (없으면 null — onError로 숨김 처리)
const photoUrl = (mlbId) =>
  `https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/${mlbId}/headshot/67/current`;

function useStatsAPI(mlbId) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const cache = useRef({});

  useEffect(() => {
    if (!mlbId) return;
    if (cache.current[mlbId]) { setData(cache.current[mlbId]); return; }
    setLoading(true);
    setData(null);

    const parseStats = (json, seasonOverride) => {
      const person = json?.people?.[0];
      if (!person) return null;
      let pitching = null, hitting = null;
      for (const sg of (person.stats || [])) {
        const splits = sg.splits || [];
        const yr = splits[0];
        if (!yr) continue;
        if (sg.group?.displayName === "pitching") pitching = { stat: yr.stat, season: seasonOverride || yr.season, team: yr.team?.name };
        if (sg.group?.displayName === "hitting")  hitting  = { stat: yr.stat, season: seasonOverride || yr.season, team: yr.team?.name };
      }
      return { person, pitching, hitting };
    };

    const base = `https://statsapi.mlb.com/api/v1/people/${mlbId}`;

    Promise.all([
      // 2025 MLB 정규시즌
      fetch(`${base}?hydrate=stats(group=[hitting,pitching],type=season,season=2025),currentTeam`).then(r=>r.json()).catch(()=>null),
      // 2024 MLB 정규시즌
      fetch(`${base}?hydrate=stats(group=[hitting,pitching],type=season,season=2024)`).then(r=>r.json()).catch(()=>null),
      // WBC 2026 (sportId=51: 국제대회)
      fetch(`${base}?hydrate=stats(group=[hitting,pitching],type=season,season=2026,sportId=51)`).then(r=>r.json()).catch(()=>null),
    ]).then(([j25, j24, jwbc]) => {
      const d25  = parseStats(j25,  "2025");
      const d24  = parseStats(j24,  "2024");
      const dwbc = parseStats(jwbc, "2026 WBC");
      const person = d25?.person || d24?.person || dwbc?.person;
      if (!person) return;
      const result = {
        person,
        pitching2025: d25?.pitching   || null,
        hitting2025:  d25?.hitting    || null,
        pitching2024: d24?.pitching   || null,
        hitting2024:  d24?.hitting    || null,
        pitchingWBC:  dwbc?.pitching  || null,
        hittingWBC:   dwbc?.hitting   || null,
        // 하위 호환
        pitching: d25?.pitching || d24?.pitching || null,
        hitting:  d25?.hitting  || d24?.hitting  || null,
      };
      cache.current[mlbId] = result;
      setData(result);
    }).finally(() => setLoading(false));
  }, [mlbId]);

  return { data, loading };
}

// ── 뉴스 검색 훅 (Anthropic API + web_search tool) ───────────────────────────
function usePlayerNews(player) {
  const [articles, setArticles] = useState(null); // [{title, url, source, date}]
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const cache = useRef({});

  const fetch_news = async () => {
    if (!player) return;
    const key = player.mlbId || player.name;
    if (cache.current[key]) { setArticles(cache.current[key]); return; }
    setLoading(true);
    setError(null);

    const query = `"${player.name}" baseball WBC 2026 OR MLB 2025 news`;

    const prompt = `Search for recent news articles about the baseball player "${player.name}" (${player.team}, ${player.pos}).
Focus on: WBC 2026 performance, recent MLB/international news from 2025-2026.

Return ONLY a JSON array (no markdown, no explanation) of up to 6 articles like:
[{"title":"Article headline","url":"https://...","source":"MLB.com","date":"2026-03-10"}]

Rules:
- Only include articles clearly about this specific player
- Prefer MLB.com, ESPN, The Athletic, Sports Illustrated, AP, Reuters
- If fewer than 6 real results exist, return fewer — never fabricate URLs
- date format: YYYY-MM-DD`;

    try {
      const response = await fetch("/api/anthropic", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          
        },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          tools: [{ type: "web_search_20250305", name: "web_search" }],
          messages: [{ role: "user", content: prompt }],
        }),
      });

      const data = await response.json();

      // content 블록에서 text만 합치기
      const fullText = (data.content || [])
        .filter(b => b.type === "text")
        .map(b => b.text)
        .join("\n");

      // JSON 배열 파싱
      const match = fullText.match(/\[[\s\S]*\]/);
      if (!match) throw new Error("no JSON array");
      const parsed = JSON.parse(match[0]);

      // 빈 배열이어도 캐시에 저장
      cache.current[key] = parsed;
      setArticles(parsed);
    } catch (e) {
      setError("기사를 불러오지 못했습니다.");
      setArticles([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (player) fetch_news();
  }, [player?.mlbId, player?.name]);

  return { articles, loading, error, refetch: fetch_news };
}

// ── 유틸 ────────────────────────────────────────────────────────────────────
const fmt3 = v => v != null ? Number(v).toFixed(3) : "-";
const fmt2 = v => v != null ? Number(v).toFixed(2) : "-";
const fmt1 = v => v != null ? Number(v).toFixed(1) : "-";

function StatBadge({ label, value, highlight }) {
  return (
    <div style={{textAlign:"center",padding:"7px 4px",background:highlight?"rgba(245,158,11,0.15)":"#0d1b3e",borderRadius:8,border:highlight?"1px solid rgba(245,158,11,0.4)":"1px solid transparent"}}>
      <div style={{fontSize:10,color:"#64748b",marginBottom:2}}>{label}</div>
      <div style={{fontSize:17,fontWeight:900,color:highlight?"#f59e0b":"#e2e8f0",lineHeight:1}}>{value??"-"}</div>
    </div>
  );
}

// ── WBC 성적 카드 (API 실시간 우선, CSV fallback) ───────────────────────────
function WBCStatsCard({ player, apiWBC }) {
  const p = player.isPitcher;

  // API에서 받은 WBC 성적 우선
  const liveP = apiWBC?.pitchingWBC;
  const liveH = apiWBC?.hittingWBC;
  const liveSt = p ? liveP?.stat : liveH?.stat;

  // CSV fallback
  const ws = player.wbcStats || {};
  const csvHasStats = p ? (ws.G > 0) : (ws.PA > 0);

  if (liveSt) {
    // API 실시간 데이터
    return (
      <div>
        <div style={{background:"#b45309",color:"#fff",borderRadius:"10px 10px 0 0",padding:"9px 16px",fontWeight:800,fontSize:13,display:"flex",alignItems:"center",gap:8}}>
          🏆 WBC 2026 성적
          <span style={{fontSize:10,fontWeight:400,opacity:.8,background:"rgba(255,255,255,0.15)",borderRadius:4,padding:"1px 6px"}}>📡 실시간</span>
        </div>
        <div style={{background:"#0d1b3e",borderRadius:"0 0 10px 10px",padding:14,display:"grid",gridTemplateColumns:"repeat(8,1fr)",gap:6}}>
          {p ? <>
            <StatBadge label="G"    value={liveSt.gamesPitched}/>
            <StatBadge label="GS"   value={liveSt.gamesStarted}/>
            <StatBadge label="IP"   value={fmt1(liveSt.inningsPitched)}/>
            <StatBadge label="ERA"  value={fmt2(liveSt.era)}  highlight={parseFloat(liveSt.era)<3}/>
            <StatBadge label="WHIP" value={fmt2(liveSt.whip)} highlight={parseFloat(liveSt.whip)<1}/>
            <StatBadge label="K%"   value={liveSt.strikeoutPercentage ? liveSt.strikeoutPercentage+"%" : liveSt.strikeOuts&&liveSt.battersFaced ? (liveSt.strikeOuts/liveSt.battersFaced*100).toFixed(1)+"%" : "-"} highlight/>
            <StatBadge label="BB%"  value={liveSt.walkPercentage ? liveSt.walkPercentage+"%" : liveSt.baseOnBalls&&liveSt.battersFaced ? (liveSt.baseOnBalls/liveSt.battersFaced*100).toFixed(1)+"%" : "-"}/>
            <StatBadge label="SO"   value={liveSt.strikeOuts}/>
          </> : <>
            <StatBadge label="PA"  value={liveSt.plateAppearances}/>
            <StatBadge label="H"   value={liveSt.hits}/>
            <StatBadge label="HR"  value={liveSt.homeRuns}  highlight={liveSt.homeRuns>0}/>
            <StatBadge label="RBI" value={liveSt.rbi}/>
            <StatBadge label="AVG" value={fmt3(liveSt.avg)} highlight={parseFloat(liveSt.avg)>0.3}/>
            <StatBadge label="OBP" value={fmt3(liveSt.obp)}/>
            <StatBadge label="BB"  value={liveSt.baseOnBalls}/>
            <StatBadge label="SB"  value={liveSt.stolenBases} highlight={liveSt.stolenBases>0}/>
          </>}
        </div>
      </div>
    );
  }

  // CSV fallback
  if (!csvHasStats) return (
    <div style={{background:"#162040",borderRadius:12,padding:"12px 16px",color:"#475569",fontSize:13}}>
      🏟️ WBC 출전 기록 없음
    </div>
  );
  return (
    <div>
      <div style={{background:"#b45309",color:"#fff",borderRadius:"10px 10px 0 0",padding:"9px 16px",fontWeight:800,fontSize:13,display:"flex",alignItems:"center",gap:8}}>
        🏆 WBC 2026 성적
        <span style={{fontSize:10,fontWeight:400,opacity:.8,background:"rgba(255,255,255,0.15)",borderRadius:4,padding:"1px 6px"}}>CSV</span>
      </div>
      <div style={{background:"#0d1b3e",borderRadius:"0 0 10px 10px",padding:14,display:"grid",gridTemplateColumns:"repeat(8,1fr)",gap:6}}>
        {p ? <>
          <StatBadge label="G"    value={ws.G}/>
          <StatBadge label="GS"   value={ws.GS}/>
          <StatBadge label="IP"   value={ws.IP}/>
          <StatBadge label="ERA"  value={ws.ERA!=null?fmt2(ws.ERA):"-"} highlight={ws.ERA!=null&&ws.ERA<3}/>
          <StatBadge label="WHIP" value={ws.WHIP!=null?fmt2(ws.WHIP):"-"} highlight={ws.WHIP!=null&&ws.WHIP<1}/>
          <StatBadge label="K%"   value={ws.Kpct!=null?ws.Kpct+"%":"-"} highlight={ws.Kpct>25}/>
          <StatBadge label="BB%"  value={ws.BBpct!=null?ws.BBpct+"%":"-"}/>
          <StatBadge label="SO"   value={ws.SO}/>
        </> : <>
          <StatBadge label="PA"  value={ws.PA}/>
          <StatBadge label="H"   value={ws.H}/>
          <StatBadge label="HR"  value={ws.HR} highlight={ws.HR>0}/>
          <StatBadge label="RBI" value={ws.RBI}/>
          <StatBadge label="AVG" value={ws.AVG!=null?fmt3(ws.AVG):"-"} highlight={ws.AVG>0.3}/>
          <StatBadge label="OBP" value={ws.OBP!=null?fmt3(ws.OBP):"-"}/>
          <StatBadge label="BB"  value={ws.BB}/>
          <StatBadge label="SB"  value={ws.SB} highlight={ws.SB>0}/>
        </>}
      </div>
    </div>
  );
}

// ── MLB StatsAPI 실시간 성적 (2025 + 2024) ───────────────────────────────────
function SeasonStatsBlock({ stat, season, team, isPitcher, isLatest }) {
  if (!stat) return (
    <div style={{background:"#0a1628",borderRadius:10,padding:"10px 14px",color:"#334155",fontSize:12,textAlign:"center"}}>
      {season} 시즌 데이터 없음
    </div>
  );
  return (
    <div>
      <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
        <span style={{fontWeight:800,fontSize:12,color:isLatest?"#60a5fa":"#475569"}}>{season} 시즌</span>
        {team && <span style={{fontSize:11,color:"#334155"}}>{team}</span>}
        {isLatest && <span style={{background:"rgba(96,165,250,0.15)",color:"#60a5fa",borderRadius:5,padding:"1px 6px",fontSize:10,fontWeight:700}}>최신</span>}
      </div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:5}}>
        {isPitcher ? <>
          <StatBadge label="ERA"    value={fmt2(stat.era)}           highlight={parseFloat(stat.era)<3}/>
          <StatBadge label="WHIP"   value={fmt2(stat.whip)}          highlight={parseFloat(stat.whip)<1}/>
          <StatBadge label="W-L"    value={`${stat.wins||0}-${stat.losses||0}`}/>
          <StatBadge label="SO"     value={stat.strikeOuts}/>
          <StatBadge label="G"      value={stat.gamesPitched}/>
          <StatBadge label="IP"     value={fmt1(stat.inningsPitched)}/>
          <StatBadge label="BB"     value={stat.baseOnBalls}/>
          <StatBadge label="HR허용" value={stat.homeRuns}/>
        </> : <>
          <StatBadge label="AVG" value={fmt3(stat.avg)}  highlight={parseFloat(stat.avg)>0.3}/>
          <StatBadge label="HR"  value={stat.homeRuns}   highlight={parseInt(stat.homeRuns)>30}/>
          <StatBadge label="RBI" value={stat.rbi}/>
          <StatBadge label="OPS" value={fmt3(stat.ops)}  highlight={parseFloat(stat.ops)>0.9}/>
          <StatBadge label="G"   value={stat.gamesPlayed}/>
          <StatBadge label="H"   value={stat.hits}/>
          <StatBadge label="SB"  value={stat.stolenBases}/>
          <StatBadge label="BB"  value={stat.baseOnBalls}/>
        </>}
      </div>
    </div>
  );
}

function LiveStatsCard({ player, data, loading }) {
  if (!player.mlbId) return null;
  if (loading) return (
    <div style={{background:"#0d1b3e",borderRadius:12,padding:"14px 16px",color:"#475569",fontSize:13,display:"flex",alignItems:"center",gap:8}}>
      <span style={{animation:"spin 1s linear infinite",display:"inline-block"}}>⏳</span> MLB 성적 불러오는 중...
    </div>
  );
  if (!data) return null;

  const p = player.isPitcher;
  const st25 = p ? data.pitching2025 : data.hitting2025;
  const st24 = p ? data.pitching2024 : data.hitting2024;
  if (!st25 && !st24) return null;

  return (
    <div style={{background:"#0d1b3e",borderRadius:12,overflow:"hidden",border:"1px solid rgba(21,101,192,0.3)"}}>
      <div style={{background:"#1565c0",color:"#fff",padding:"9px 16px",fontWeight:800,fontSize:13,display:"flex",alignItems:"center",gap:6}}>
        📡 MLB 정규시즌 성적
        <span style={{fontSize:10,opacity:.7,fontWeight:400,marginLeft:4}}>{data.person?.currentTeam?.name || data.person?.fullName}</span>
      </div>
      <div style={{padding:14,display:"flex",flexDirection:"column",gap:14}}>
        <SeasonStatsBlock stat={st25?.stat} season="2025" team={st25?.team} isPitcher={p} isLatest={true}/>
        <div style={{borderTop:"1px solid rgba(255,255,255,0.05)"}}/>
        <SeasonStatsBlock stat={st24?.stat} season="2024" team={st24?.team} isPitcher={p} isLatest={false}/>
      </div>
    </div>
  );
}

// ── 뉴스 패널 ────────────────────────────────────────────────────────────────
const SOURCE_COLORS = {
  "MLB.com":"#002d72","ESPN":"#d50a0a","The Athletic":"#1a1a1a",
  "Sports Illustrated":"#c8102e","AP":"#333","Reuters":"#ff8000",
};

function NewsPanel({ player }) {
  const { articles, loading, error, refetch } = usePlayerNews(player);

  return (
    <div style={{background:"#0c1a2e",borderRadius:12,overflow:"hidden",border:"1px solid rgba(96,165,250,0.2)"}}>
      <div style={{background:"linear-gradient(90deg,#1e3a5f,#1e293b)",padding:"10px 16px",display:"flex",alignItems:"center",gap:8}}>
        <span style={{fontSize:14}}>📰</span>
        <span style={{fontWeight:800,fontSize:13,color:"#93c5fd",letterSpacing:.5}}>관련 기사</span>
        {loading && <span style={{fontSize:11,color:"#475569",marginLeft:4,animation:"pulse 1s infinite"}}>검색 중...</span>}
        {!loading && articles && (
          <span style={{fontSize:11,color:"#334155",marginLeft:4}}>{articles.length}건</span>
        )}
        <button onClick={refetch} title="새로고침"
          style={{marginLeft:"auto",background:"transparent",border:"none",color:"#334155",cursor:"pointer",fontSize:14,padding:2,lineHeight:1}}
        >↻</button>
      </div>

      <div style={{padding:"10px 14px",display:"flex",flexDirection:"column",gap:0}}>
        {loading && (
          <div style={{display:"flex",flexDirection:"column",gap:8,padding:"4px 0"}}>
            {[1,2,3].map(i=>(
              <div key={i} style={{height:52,borderRadius:8,background:"rgba(255,255,255,0.03)",animation:"pulse 1.5s infinite",animationDelay:`${i*0.15}s`}}/>
            ))}
          </div>
        )}

        {!loading && error && (
          <div style={{padding:"12px 0",color:"#475569",fontSize:13,textAlign:"center"}}>{error}</div>
        )}

        {!loading && articles?.length === 0 && (
          <div style={{padding:"12px 0",color:"#334155",fontSize:13,textAlign:"center"}}>
            관련 기사를 찾지 못했습니다.
          </div>
        )}

        {!loading && articles?.map((art, i) => {
          const srcColor = SOURCE_COLORS[art.source] || "#334155";
          const domain = (() => { try { return new URL(art.url).hostname.replace("www.",""); } catch{ return art.source||""; } })();
          return (
            <a key={i} href={art.url} target="_blank" rel="noreferrer"
              style={{display:"block",padding:"11px 0",borderBottom:i<articles.length-1?"1px solid rgba(255,255,255,0.04)":"none",textDecoration:"none",transition:"all 0.1s"}}
              onMouseEnter={e=>{e.currentTarget.style.paddingLeft="6px";}}
              onMouseLeave={e=>{e.currentTarget.style.paddingLeft="0px";}}
            >
              <div style={{fontWeight:600,fontSize:13,color:"#cbd5e1",lineHeight:1.45,marginBottom:5}}>
                {art.title}
              </div>
              <div style={{display:"flex",alignItems:"center",gap:8}}>
                <span style={{fontSize:10,fontWeight:700,color:"#fff",background:srcColor,borderRadius:4,padding:"1px 6px"}}>
                  {art.source || domain}
                </span>
                <span style={{fontSize:10,color:"#334155"}}>{domain}</span>
                {art.date && <span style={{fontSize:10,color:"#334155",marginLeft:"auto"}}>{art.date}</span>}
              </div>
            </a>
          );
        })}
      </div>
    </div>
  );
}

// ── 선수 모달 ────────────────────────────────────────────────────────────────
function PlayerModal({ player, onClose }) {
  const { data, loading: statsLoading } = useStatsAPI(player?.mlbId);
  const [imgOk, setImgOk] = useState(false);
  if (!player) return null;
  const flag = TEAM_FLAG[player.team] || "🏳️";
  const photo = player.mlbId ? photoUrl(player.mlbId) : null;

  return (
    <div style={{position:"fixed",inset:0,background:"rgba(0,0,0,0.8)",zIndex:1000,display:"flex",alignItems:"center",justifyContent:"center",padding:16,backdropFilter:"blur(4px)"}} onClick={onClose}>
      <div style={{background:"linear-gradient(160deg,#1a1d2e 0%,#0f1629 100%)",borderRadius:22,maxWidth:720,width:"100%",maxHeight:"90vh",overflowY:"auto",color:"#e8eaf6",boxShadow:"0 32px 96px rgba(0,0,0,0.9)"}} onClick={e=>e.stopPropagation()}>

        {/* 헤더 */}
        <div style={{background:"linear-gradient(135deg,#0f1629,#1a2744)",borderRadius:"22px 22px 0 0",padding:24,position:"relative",borderBottom:"1px solid rgba(255,255,255,0.06)"}}>
          <button onClick={onClose} style={{position:"absolute",top:16,right:16,background:"rgba(255,255,255,0.08)",border:"none",color:"#94a3b8",width:34,height:34,borderRadius:10,cursor:"pointer",fontSize:18,display:"flex",alignItems:"center",justifyContent:"center"}}>×</button>
          <div style={{display:"flex",gap:16,alignItems:"flex-start"}}>
            {/* 선수 사진 */}
            <div style={{width:80,height:80,borderRadius:16,overflow:"hidden",flexShrink:0,background:"rgba(255,255,255,0.06)",border:"1px solid rgba(255,255,255,0.1)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:32}}>
              {photo ? (
                <>
                  <img src={photo} alt={player.name}
                    style={{width:"100%",height:"100%",objectFit:"cover",display:imgOk?"block":"none"}}
                    onLoad={()=>setImgOk(true)}
                    onError={()=>setImgOk(false)}
                  />
                  {!imgOk && flag}
                </>
              ) : flag}
            </div>
            <div style={{flex:1}}>
              <div style={{display:"flex",gap:8,alignItems:"center",flexWrap:"wrap",marginBottom:5}}>
                <h2 style={{margin:0,fontSize:21,fontWeight:900,color:"#fff",letterSpacing:"-0.5px"}}>{player.name}</h2>
                <span style={{background:"rgba(255,255,255,0.1)",color:"#cbd5e1",borderRadius:7,padding:"2px 8px",fontSize:12,fontWeight:700}}>#{player.num}</span>
              </div>
              <div style={{color:"#64748b",fontSize:13,lineHeight:1.6}}>
                {player.pos} · 타{player.bats}/투{player.throws} · {player.h}cm/{player.w}kg · {player.age}세 ({player.dob})
              </div>
              <div style={{color:"#64748b",fontSize:13}}>{flag} {player.team} · {player.pool}</div>
              {player.mlbId && (
                <a href={`https://www.mlb.com/player/${player.mlbId}`} target="_blank" rel="noreferrer"
                  style={{display:"inline-flex",alignItems:"center",gap:5,marginTop:8,background:"rgba(21,101,192,0.3)",color:"#93c5fd",borderRadius:7,padding:"4px 10px",fontSize:12,fontWeight:700,textDecoration:"none",border:"1px solid rgba(21,101,192,0.4)"}}>
                  ⚾ MLB.com 프로필
                </a>
              )}
            </div>
          </div>
        </div>

        <div style={{padding:22,display:"flex",flexDirection:"column",gap:16}}>
          {/* WBC 2026 성적 — API 실시간 우선 */}
          <WBCStatsCard player={player} apiWBC={data}/>

          {/* MLB StatsAPI 실시간 — data를 위에서 받아 내려줌 */}
          <LiveStatsCard player={player} data={data} loading={statsLoading}/>

          {/* 관련 기사 */}
          <NewsPanel player={player}/>
        </div>
      </div>
    </div>
  );
}

// ── 선수 행 ──────────────────────────────────────────────────────────────────
function PlayerRow({ p, onClick }) {
  const ws = p.wbcStats || {};
  const hasStats = p.isPitcher ? ws.G > 0 : ws.PA > 0;
  return (
    <div onClick={()=>onClick(p)}
      style={{display:"flex",alignItems:"center",gap:10,background:"#1a1d2e",borderRadius:10,padding:"8px 12px",cursor:"pointer",transition:"all 0.12s",borderLeft:`3px solid ${p.isPitcher?"#3b82f6":"#f59e0b"}`}}
      onMouseEnter={e=>{e.currentTarget.style.background="#1e2235";e.currentTarget.style.transform="translateX(2px)";}}
      onMouseLeave={e=>{e.currentTarget.style.background="#1a1d2e";e.currentTarget.style.transform="translateX(0)";}}
    >
      <div style={{width:28,height:28,borderRadius:7,background:p.isPitcher?"rgba(59,130,246,0.2)":"rgba(245,158,11,0.2)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:13,flexShrink:0,border:`1px solid ${p.isPitcher?"rgba(59,130,246,0.3)":"rgba(245,158,11,0.3)"}`}}>
        {p.isPitcher?"🥎":"🏏"}
      </div>
      <div style={{flex:1,minWidth:0}}>
        <div style={{fontWeight:700,fontSize:13,color:"#e2e8f0",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{p.name}</div>
        <div style={{fontSize:10,color:"#475569"}}>{p.pos} · #{p.num}</div>
      </div>
      {hasStats && (
        <div style={{fontSize:11,color:"#94a3b8",textAlign:"right",flexShrink:0}}>
          {p.isPitcher
            ? <span style={{color:ws.ERA<3?"#34d399":"#94a3b8"}}>ERA {ws.ERA!=null?fmt2(ws.ERA):"-"}</span>
            : <span style={{color:ws.AVG>0.3?"#fbbf24":"#94a3b8"}}>AVG {ws.AVG!=null?fmt3(ws.AVG):"-"}</span>
          }
        </div>
      )}
    </div>
  );
}

// ── 팀 섹션 ──────────────────────────────────────────────────────────────────
function TeamSection({ teamName, players, onSelect }) {
  const [open, setOpen] = useState(true);
  const tp = players.filter(p=>p.team===teamName);
  const mlb = tp.filter(p=>p.mlbId).length;
  const hasStats = tp.filter(p=>p.isPitcher ? p.wbcStats.G>0 : p.wbcStats.PA>0).length;
  return (
    <div style={{background:"#fff",borderRadius:16,overflow:"hidden",border:"1px solid #e2e8f0"}}>
      <div onClick={()=>setOpen(!open)} style={{display:"flex",alignItems:"center",gap:10,padding:"12px 16px",cursor:"pointer",background:open?"#f8fafc":"#fff"}}>
        <span style={{fontSize:22}}>{TEAM_FLAG[teamName]||"🏳️"}</span>
        <div style={{flex:1}}>
          <div style={{fontWeight:900,fontSize:15,color:"#1e293b"}}>{teamName}</div>
          <div style={{fontSize:11,color:"#94a3b8"}}>{tp.length}명 · MLB연동 {mlb} · WBC기록 {hasStats}</div>
        </div>
        <span style={{color:"#94a3b8",fontSize:14,transform:open?"rotate(180deg)":"",transition:"transform 0.2s"}}>▼</span>
      </div>
      {open && (
        <div style={{padding:"0 10px 10px",display:"flex",flexDirection:"column",gap:4,maxHeight:500,overflowY:"auto"}}>
          {tp.map(p=><PlayerRow key={p.id} p={p} onClick={onSelect}/>)}
        </div>
      )}
    </div>
  );
}

// ── 메인 App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState("home");
  const [filterTeam, setFilterTeam] = useState("");
  const [filterPos, setFilterPos] = useState("전체");

  const allTeams = [...new Set(ALL_PLAYERS.map(p=>p.team))].sort();

  const filtered = ALL_PLAYERS.filter(p => {
    const q = searchQuery.toLowerCase();
    const matchQ = !q || p.name.toLowerCase().includes(q) || p.team.toLowerCase().includes(q);
    const matchT = !filterTeam || p.team===filterTeam;
    const matchP = filterPos==="전체" || (filterPos==="투수" && p.isPitcher) || (filterPos==="타자" && !p.isPitcher);
    return matchQ && matchT && matchP;
  });

  const mlbCount = ALL_PLAYERS.filter(p=>p.mlbId).length;
  const hasStatsCount = ALL_PLAYERS.filter(p=>p.isPitcher?(p.wbcStats.G>0):(p.wbcStats.PA>0)).length;

  return (
    <div style={{fontFamily:"'Noto Sans KR','Segoe UI',sans-serif",background:"#f1f5f9",minHeight:"100vh"}}>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        ::-webkit-scrollbar{width:5px;height:5px}
        ::-webkit-scrollbar-track{background:transparent}
        ::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}
      `}</style>

      {/* Nav */}
      <nav style={{background:"#0f1629",padding:"0 16px",height:54,display:"flex",alignItems:"center",gap:12,position:"sticky",top:0,zIndex:200,borderBottom:"1px solid rgba(255,255,255,0.06)"}}>
        <div style={{fontWeight:900,fontSize:16,color:"#fff",letterSpacing:"-0.5px",whiteSpace:"nowrap"}}>⚾ WBC 2026</div>
        <div style={{flex:1}}/>
        {[["home","🏠 홈"],["roster","📋 명단"]].map(([tab,label])=>(
          <button key={tab} onClick={()=>setActiveTab(tab)}
            style={{background:activeTab===tab?"rgba(255,255,255,0.12)":"transparent",border:activeTab===tab?"1px solid rgba(255,255,255,0.15)":"1px solid transparent",color:activeTab===tab?"#fff":"#64748b",padding:"5px 12px",borderRadius:8,cursor:"pointer",fontSize:12,fontWeight:700,whiteSpace:"nowrap"}}>
            {label}
          </button>
        ))}
        <div style={{position:"relative"}}>
          <input placeholder="🔍 선수 검색..." value={searchQuery} onChange={e=>setSearchQuery(e.target.value)}
            style={{background:"rgba(255,255,255,0.08)",border:"1px solid rgba(255,255,255,0.12)",color:"#fff",borderRadius:10,padding:"6px 12px",fontSize:13,width:180,outline:"none"}}/>
          {searchQuery && (
            <div style={{position:"absolute",top:"calc(100% + 4px)",right:0,width:340,background:"#1a1d2e",borderRadius:12,zIndex:300,boxShadow:"0 8px 32px rgba(0,0,0,0.6)",maxHeight:360,overflowY:"auto",border:"1px solid rgba(255,255,255,0.08)"}}>
              {filtered.length===0
                ? <div style={{padding:16,color:"#475569",textAlign:"center",fontSize:13}}>검색 결과 없음</div>
                : filtered.slice(0,12).map(p=>(
                  <div key={p.id} onClick={()=>{setSelectedPlayer(p);setSearchQuery("");}}
                    style={{padding:"9px 14px",cursor:"pointer",color:"#e2e8f0",borderBottom:"1px solid rgba(255,255,255,0.04)",display:"flex",gap:8,alignItems:"center"}}
                    onMouseEnter={e=>e.currentTarget.style.background="rgba(255,255,255,0.06)"}
                    onMouseLeave={e=>e.currentTarget.style.background="transparent"}
                  >
                    <span style={{fontSize:18}}>{TEAM_FLAG[p.team]||"🏳️"}</span>
                    <div style={{flex:1,minWidth:0}}>
                      <div style={{fontWeight:700,fontSize:13,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{p.name}</div>
                      <div style={{fontSize:10,color:"#475569"}}>{p.team} · {p.pos} · #{p.num}</div>
                    </div>
                    {(p.isPitcher ? p.wbcStats.G>0 : p.wbcStats.PA>0) && (
                      <span style={{fontSize:11,color:"#10b981",fontWeight:700}}>📊</span>
                    )}
                  </div>
                ))
              }
            </div>
          )}
        </div>
      </nav>

      <main style={{maxWidth:1400,margin:"0 auto",padding:"20px 14px"}}>

        {activeTab==="home" && (
          <>
            {/* 통계 카드 */}
            <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10,marginBottom:22}}>
              {[
                ["20개국","WBC 참가국","#1e40af"],
                [ALL_PLAYERS.length+"명","등록 선수","#065f46"],
                [mlbCount+"명","MLB ID 연동","#1e40af"],
                [hasStatsCount+"명","WBC 성적 보유","#92400e"],
              ].map(([v,l,c])=>(
                <div key={l} style={{background:"#1a1d2e",borderRadius:14,padding:"16px 14px",color:"#fff",borderLeft:`4px solid ${c}`}}>
                  <div style={{fontSize:24,fontWeight:900,color:"#f1f5f9",lineHeight:1}}>{v}</div>
                  <div style={{fontSize:12,color:"#64748b",marginTop:4}}>{l}</div>
                </div>
              ))}
            </div>

            {/* Pool별 팀 */}
            {Object.entries(WBC_POOLS).map(([poolName,poolData])=>(
              <div key={poolName} style={{marginBottom:26}}>
                <h2 style={{fontSize:16,fontWeight:900,marginBottom:12,color:"#1e293b",display:"flex",alignItems:"center",gap:8}}>
                  <span style={{background:"#1e40af",color:"#fff",borderRadius:8,padding:"3px 10px",fontSize:13}}>{poolName}</span>
                  <span style={{fontSize:13,color:"#94a3b8",fontWeight:400}}>📍 {poolData.location}</span>
                </h2>
                <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(280px,1fr))",gap:10}}>
                  {poolData.teams.map(team=><TeamSection key={team} teamName={team} players={ALL_PLAYERS} onSelect={setSelectedPlayer}/>)}
                </div>
              </div>
            ))}
          </>
        )}

        {activeTab==="roster" && (
          <div>
            <div style={{display:"flex",gap:10,alignItems:"center",marginBottom:18,flexWrap:"wrap"}}>
              <h1 style={{fontSize:18,fontWeight:900,margin:0,color:"#1e293b"}}>전체 명단</h1>
              <select value={filterTeam} onChange={e=>setFilterTeam(e.target.value)}
                style={{background:"#1a1d2e",color:"#e2e8f0",border:"1px solid #334155",borderRadius:8,padding:"5px 10px",fontSize:13}}>
                <option value="">전체 국가</option>
                {allTeams.map(t=><option key={t}>{t}</option>)}
              </select>
              <select value={filterPos} onChange={e=>setFilterPos(e.target.value)}
                style={{background:"#1a1d2e",color:"#e2e8f0",border:"1px solid #334155",borderRadius:8,padding:"5px 10px",fontSize:13}}>
                {["전체","투수","타자"].map(v=><option key={v}>{v}</option>)}
              </select>
              <span style={{fontSize:13,color:"#64748b"}}>{filtered.length}명</span>
            </div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(300px,1fr))",gap:6}}>
              {filtered.map(p=>(
                <div key={p.id} onClick={()=>setSelectedPlayer(p)}
                  style={{background:"#1a1d2e",borderRadius:11,padding:"10px 14px",cursor:"pointer",display:"flex",alignItems:"center",gap:10,color:"#e2e8f0",borderLeft:`3px solid ${p.isPitcher?"#3b82f6":"#f59e0b"}`,transition:"all 0.1s"}}
                  onMouseEnter={e=>{e.currentTarget.style.background="#1e2235";}}
                  onMouseLeave={e=>{e.currentTarget.style.background="#1a1d2e";}}
                >
                  <span style={{fontSize:18,flexShrink:0}}>{TEAM_FLAG[p.team]||"🏳️"}</span>
                  <div style={{flex:1,minWidth:0}}>
                    <div style={{fontWeight:700,fontSize:13,display:"flex",gap:6,alignItems:"center"}}>
                      <span style={{overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{p.name}</span>
                      <span style={{fontSize:10,color:"#475569",flexShrink:0}}>#{p.num}</span>
                    </div>
                    <div style={{fontSize:10,color:"#475569"}}>{p.pos} · {p.team}</div>
                  </div>
                  <div style={{textAlign:"right",flexShrink:0,fontSize:11}}>
                    {(p.isPitcher ? p.wbcStats.G>0 : p.wbcStats.PA>0) ? (
                      <span style={{color:"#10b981",fontWeight:700}}>
                        {p.isPitcher ? `ERA ${p.wbcStats.ERA!=null?fmt2(p.wbcStats.ERA):"-"}` : `AVG ${p.wbcStats.AVG!=null?fmt3(p.wbcStats.AVG):"-"}`}
                      </span>
                    ) : <span style={{color:"#334155"}}>-</span>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      <div style={{background:"#0f1629",color:"#334155",padding:"10px 20px",fontSize:11,textAlign:"center",borderTop:"1px solid rgba(255,255,255,0.04)"}}>
        출처: WBC 2026 공식 통계 CSV · MLB StatsAPI · Claude AI 코멘트 · 총 {ALL_PLAYERS.length}명 (파나마 제외 CSV 기준)
      </div>

      {selectedPlayer && <PlayerModal player={selectedPlayer} onClose={()=>setSelectedPlayer(null)}/>}
    </div>
  );
}

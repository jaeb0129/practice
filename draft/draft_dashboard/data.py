import pandas as pd
import numpy as np

from utils import load_raw

data_pit, data_bat = load_raw()

def tracking_pitcher():
    hs25 = data_pit
    
    hs25['Zone'] = ( (hs25["PlateLocHeight"] <= 1.07) &
                     (hs25["PlateLocHeight"] >= 0.43) &
                     (hs25["PlateLocSide"] <= 0.3) &
                     (hs25["PlateLocSide"] >= -0.3)).astype(int)
    
    hs25["is_pa"] = ((hs25["PlayResult"].isin(['Single', 'Double', 'Triple', 'HomeRun', 'Error',
                                               'FieldersChoice', 'Out', 'Sacrifice'])) | (hs25["KorBB"].isin(['Walk', 'Strikeout']))).astype(int)
    
    # ── 1. 투수별 집계 ──────────────────────────────────────
    swing_events = ["StrikeSwinging", "InPlay", "FoulBall"]
    strike_events = ["StrikeCalled", "StrikeSwinging", "InPlay", "FoulBall"]
    csw_events = ["StrikeCalled", "StrikeSwinging"]

    # 기본 플래그
    hs25["is_swing"] = hs25["PitchCall"].isin(swing_events).astype(int)
    hs25["is_whiff"] = (hs25["PitchCall"] == "StrikeSwinging").astype(int)
    hs25["is_strike"] = hs25["PitchCall"].isin(strike_events).astype(int)
    hs25["is_lk"] = (hs25["PitchCall"] == "StrikeCalled").astype(int)
    hs25["is_csw"] = (hs25["PitchCall"].isin(csw_events)).astype(int)
                      
    hs25["is_first_pitch"] = ((hs25["Strikes"] == 0) & (hs25["Balls"] == 0)).astype(int)
    hs25["is_first_pitch_s"] = ((hs25["Strikes"] == 0) & (hs25["Balls"] == 0) & (hs25["PitchCall"].isin(strike_events))).astype(int)
    
    hs25["is_bb"] = (hs25["KorBB"] == "Walk").astype(int)
    hs25["is_so"] = (hs25["KorBB"] == "Strikeout").astype(int)

    # 존 여부 (이미 Zone 컬럼 있다고 가정: 1=존, 0=볼)
    hs25["is_zone"] = (hs25["Zone"] == 1).astype(int)
    hs25["is_outzone"] = (hs25["Zone"] == 0).astype(int)

    # 존 스윙 / 볼존 스윙 (Chase)
    hs25["is_z_swing"] = ((hs25["is_zone"] == 1) & (hs25["is_swing"] == 1)).astype(int)
    hs25["is_o_swing"] = ((hs25["is_outzone"] == 1) & (hs25["is_swing"] == 1)).astype(int)

    
    hs25["is_z_contact"] = ((hs25["is_zone"] == 1) & (hs25["is_swing"] == 1) & (hs25["is_whiff"] == 0)).astype(int)
    hs25["is_o_contact"] = ((hs25["is_zone"] == 0) & (hs25["is_swing"] == 1) & (hs25["is_whiff"] == 0)).astype(int)
    
    hs25["is_2s_swing"] = ((hs25["Strikes"] == 2) & (hs25["is_swing"] == 1)).astype(int)
    hs25["is_2s_whiff"] = ((hs25["Strikes"] == 2) & (hs25["is_swing"] == 1) & (hs25["is_whiff"] == 1)).astype(int)
    
    hs25["is_2s_pitch"] = ((hs25["Strikes"] == 2)).astype(int)
    hs25["is_2s_so"] = ((hs25["Strikes"] == 2) & (hs25["KorBB"] == "Strikeout")).astype(int)
    
    hs25["is_lk_so"] = ((hs25["Strikes"] == 2) & (hs25["PitchCall"] == "StrikeCalled")).astype(int)
    
    result = (
    hs25.groupby(["PitcherId", "Pitcher", "PitcherTeam", "PitcherThrows"])
    .agg(
        투구수=("PitchCall", "count"),
        타석=("is_pa", "sum"),
        스트라이크 = ("is_strike", "sum"),
        csw = ("is_csw", "sum"),
        
        볼넷 = ("is_bb", "sum"),
        삼진 = ("is_so", "sum"),

        반응=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),

        zone_pitches=("is_zone", "sum"),
        z_swings=("is_z_swing", "sum"),
        z_contacts=("is_z_contact", "sum"),

        ozone_pitches=("is_outzone", "sum"),
        o_swings=("is_o_swing", "sum"),
        o_contacts=("is_o_contact", "sum"),
        
        swings_2s=("is_2s_swing", "sum"),
        whiffs_2s=("is_2s_whiff", "sum"),
        
        lk_so=("is_lk_so", "sum"),
        
        pit_2s=("is_2s_pitch", "sum"),
        so_2s=("is_2s_so", "sum"),
        
        first_pit=("is_first_pitch", "sum"),
        first_pit_s=("is_first_pitch_s", "sum"),
        
    )
    .reset_index()
    )
    result["타석당투구수"] = (result["투구수"] / result["타석"]).round(2)
    
    result['Zone%'] = (result["zone_pitches"] / result["투구수"] * 100).round(1)
    result['초구S%'] = (result["first_pit_s"] / result["first_pit"] * 100).round(1)
    result['S%'] = (result["스트라이크"] / result["투구수"] * 100).round(1)
    result['CSW%'] = (result["스트라이크"] / result["투구수"] * 100).round(1)
    
    result['BB%'] = (result["볼넷"] / result["타석"] *100).round(1)
    result['K%'] = (result["삼진"] / result["타석"] * 100).round(1)
    result["2S삼진결정%"] = (result["so_2s"] / result["pit_2s"] * 100).round(1)
    result["루킹삼진%"] = (result["lk_so"] / result["삼진"] * 100).round(1)
    
    result["헛스윙%"] = (result["whiffs"] / result["반응"] * 100).round(1)
    
    result["반응%"] = (result["반응"] / result["투구수"] * 100).round(1)
    result["존반응%"] = (result["z_swings"] / result["zone_pitches"] * 100).round(1)
    result["존밖반응%"] = (result["o_swings"] / result["ozone_pitches"] * 100).round(1)

    
    # ── 2. 투수별 × 패스트볼 구속 집계 ──────────────────────────────────────
    filtered_df = hs25[hs25["AutoPitchType"].isin(['Four-Seam', 'Sinker'])]

    by_pitcher_fastball = (
    filtered_df.groupby(["PitcherId", "Pitcher", "PitcherTeam"])
    .agg(
        직구최고구속=("RelSpeed", lambda x: round(x.max(), 1)),
        직구평균구속=("RelSpeed", lambda x: x.mean().round(1))
    )
    .reset_index()
    )
    
    stats = pd.merge(result, by_pitcher_fastball, on=["PitcherId", "Pitcher", "PitcherTeam"], how="left")
    
    stats = stats.loc[:,["PitcherId", "Pitcher", "PitcherTeam", "PitcherThrows", "투구수", "타석", "타석당투구수", "Zone%", "초구S%", "S%", 'CSW%', "BB%", "K%", "2S삼진결정%", "루킹삼진%", "헛스윙%", "반응%", "존반응%", "존밖반응%", "직구최고구속", "직구평균구속"]]
    
    stats["PitcherThrows"] = stats["PitcherThrows"].map({"Right": "우투", "Left": "좌투"})
    stats["PitcherThrows"] = pd.Categorical(
        stats["PitcherThrows"], categories=["우투", "좌투"], ordered=True)

    return stats

def tracking_batter_discipline():
    hs25 = data_bat
    
    hs25['Zone'] = ( (hs25["PlateLocHeight"] <= 1.07) &
                     (hs25["PlateLocHeight"] >= 0.43) &
                     (hs25["PlateLocSide"] <= 0.3) &
                     (hs25["PlateLocSide"] >= -0.3)).astype(int)
    
    hs25["is_pa"] = ((hs25["PlayResult"].isin(['Single', 'Double', 'Triple', 'HomeRun', 'Error',
                                               'FieldersChoice', 'Out', 'Sacrifice'])) | (hs25["KorBB"].isin(['Walk', 'Strikeout']))).astype(int)

    # 스윙 정의
    swing_events = ["StrikeSwinging", "InPlay", "FoulBall"]

    # 기본 플래그
    hs25["is_swing"] = hs25["PitchCall"].isin(swing_events).astype(int)
    hs25["is_whiff"] = (hs25["PitchCall"] == "StrikeSwinging").astype(int)
    
    hs25["is_bb"] = (hs25["KorBB"] == "Walk").astype(int)
    hs25["is_so"] = (hs25["KorBB"] == "Strikeout").astype(int)

    # 존 여부 (이미 Zone 컬럼 있다고 가정: 1=존, 0=볼)
    hs25["is_zone"] = (hs25["Zone"] == 1).astype(int)
    hs25["is_outzone"] = (hs25["Zone"] == 0).astype(int)

    # 존 스윙 / 볼존 스윙 (Chase)
    hs25["is_z_swing"] = ((hs25["is_zone"] == 1) & (hs25["is_swing"] == 1)).astype(int)
    hs25["is_o_swing"] = ((hs25["is_outzone"] == 1) & (hs25["is_swing"] == 1)).astype(int)

    # 컨택 (스윙했는데 헛스윙 아님)
    hs25["is_contact"] = ((hs25["is_swing"] == 1) & (hs25["is_whiff"] == 0)).astype(int)
    
    hs25["is_z_contact"] = ((hs25["is_zone"] == 1) & (hs25["is_swing"] == 1) & (hs25["is_whiff"] == 0)).astype(int)
    hs25["is_o_contact"] = ((hs25["is_zone"] == 0) & (hs25["is_swing"] == 1) & (hs25["is_whiff"] == 0)).astype(int)
    
    hs25["is_2s_swing"] = ((hs25["Strikes"] == 2) & (hs25["is_swing"] == 1)).astype(int)
    hs25["is_2s_contact"] = ((hs25["Strikes"] == 2) & (hs25["is_swing"] == 1) & (hs25["is_whiff"] == 0)).astype(int)
    
    hs25["is_145_swing"] = ((hs25["RelSpeed"] >= 145) & (hs25["is_swing"] == 1)).astype(int)
    hs25["is_145_contact"] = ((hs25["RelSpeed"] >= 145) & (hs25["is_swing"] == 1) & (hs25["is_whiff"] == 0)).astype(int)
    
    hs25["is_first_pitch"] = ((hs25["Strikes"] == 0) & (hs25["Balls"] == 0)).astype(int)
    hs25["is_first_pitch_swing"] = ((hs25["Strikes"] == 0) & (hs25["Balls"] == 0) & (hs25["is_swing"] == 1)).astype(int)
    
    
    result = (
    hs25.groupby(["BatterId", "Batter", "BatterTeam", "BatterSide"])
    .agg(
        투구수=("PitchCall", "count"),
        타석=("is_pa", "sum"),

        반응=("is_swing", "sum"),
        초구반응=("is_first_pitch_swing", "sum"),
        초구=("is_first_pitch", "sum"),
        
        whiffs=("is_whiff", "sum"),
        contacts=("is_contact", "sum"),
        
        볼넷 = ("is_bb", "sum"),
        삼진 = ("is_so", "sum"),

        zone_pitches=("is_zone", "sum"),
        z_swings=("is_z_swing", "sum"),
        z_contacts=("is_z_contact", "sum"),

        ozone_pitches=("is_outzone", "sum"),
        o_swings=("is_o_swing", "sum"),
        o_contacts=("is_o_contact", "sum"),
        
        swings_2s=("is_2s_swing", "sum"),
        contacts_2s=("is_2s_contact", "sum"),
        
        swings_145=("is_145_swing", "sum"),
        contacts_145=("is_145_contact", "sum")
    )
    .reset_index()
    )
    
    result["타석당투구수"] = (result["투구수"] / result["타석"]).round(2)
    
    result['BB%'] = (result["볼넷"] / result["타석"] *100).round(1)
    result['K%'] = (result["삼진"] / result["타석"] * 100).round(1)
    
    result["반응%"] = (result["반응"] / result["투구수"] * 100).round(1)
    result["초구반응%"] = (result["초구반응"] / result["초구"] * 100).round(1)
    result["헛스윙%"] = (result["whiffs"] / result["반응"] * 100).round(1)
    result["컨택%"] = (result["contacts"] / result["반응"] * 100).round(1)
    
    result["컨택%(2S)"] = (result["contacts_2s"] / result["swings_2s"] * 100).round(1)
    result["컨택%(145이상)"] = (result["contacts_145"] / result["swings_145"] * 100).round(1)

    result["존반응%"] = (result["z_swings"] / result["zone_pitches"] * 100).round(1)
    result["존밖반응%"] = (result["o_swings"] / result["ozone_pitches"] * 100).round(1)

    result["존컨택%"] = (result["z_contacts"] / result["z_swings"] * 100).round(1)
    result["존밖컨택%"] = (result["o_contacts"] / result["o_swings"] * 100).round(1)
    
    stats = result.loc[:,["BatterId", "Batter", "BatterTeam", "BatterSide", "투구수", "타석", "타석당투구수", "BB%", "K%", "초구반응%", "반응%", "헛스윙%", "컨택%", "컨택%(2S)", "컨택%(145이상)", "존반응%", "존밖반응%", "존컨택%", "존밖컨택%"]]
    
    stats = stats.rename(columns={
        "Batter":     "선수명",
        "BatterTeam": "학교",
    })
    
    stats["BatterSide"] = stats["BatterSide"].map({"Right": "우타", "Left": "좌타"})
    stats["BatterSide"] = pd.Categorical(
    stats["BatterSide"], categories=["우타", "좌타"], ordered=True)
    
    return stats

def tracking_batter_hitrack():
    hs25 = data_bat

    hs25["is_pa"] = ((hs25["PlayResult"].isin(['Single', 'Double', 'Triple', 'HomeRun', 'Error',
                                               'FieldersChoice', 'Out', 'Sacrifice'])) | (hs25["KorBB"].isin(['Walk', 'Strikeout']))).astype(int)
    hs25["is_inplay"] = ((hs25["PlayResult"].isin(['Single', 'Double', 'Triple', 'HomeRun', 'Error',
                                               'FieldersChoice', 'Out']) & hs25["ExitSpeed"].notna())).astype(int)

    hs25["is_exitspeed"] = hs25.loc[hs25['ExitSpeed'].notna() & hs25['Angle'].notna() & hs25["is_inplay"].astype(bool), "ExitSpeed"]
    hs25["is_angle"]     = hs25.loc[hs25['ExitSpeed'].notna() & hs25['Angle'].notna() & hs25["is_inplay"].astype(bool), "Angle"]
    hs25["is_distance"]     = hs25.loc[hs25['ExitSpeed'].notna() & hs25['Angle'].notna() & hs25["is_inplay"].astype(bool), "Distance"]
    hs25["is_sweetspot"]   = ((hs25['Angle'] >= 8) & (hs25['Angle'] <= 32) & (hs25["is_inplay"] == 1)).astype(int)

    hs25["is_hardhit"] = ((hs25['ExitSpeed'] >= 152.88768) & (hs25["is_inplay"] == 1)).astype(int)

    hs25['is_gb']     = (hs25['TaggedHitType'] == 'GroundBall').astype(int)
    hs25['is_fb']     = (hs25['TaggedHitType'].isin(['FlyBall', 'Popup'])).astype(int)
    hs25['is_ld']     = (hs25['TaggedHitType'] == 'LineDrive').astype(int)
    hs25['is_bunt']   = (hs25['TaggedHitType'] == 'Bunt').astype(int)
    hs25['is_hitype'] = (hs25['TaggedHitType'].notna() & (hs25["PlayResult"].isin(['Single', 'Double', 'Triple', 'HomeRun', 'Error',
                                               'FieldersChoice', 'Out', 'Sacrifice']))).astype(int)

    # ── ICR / BARREL 플래그 ───────────────────────────────────────────────────
    es  = pd.to_numeric(hs25["ExitSpeed"], errors="coerce")
    ang = pd.to_numeric(hs25["Angle"],     errors="coerce")
    ev  = es / 1.609344   # km/h → mph
    ip  = hs25["is_inplay"] == 1

    cond6  = ip & (ev*1.5 - ang >= 117) & (ev + ang >= 124) & (ev >= 98) & ang.between(4, 50)
    cond5  = ip & (ev*1.5 - ang >= 111) & (ev + ang >= 119) & (ev >= 95) & ang.between(0, 52)
    cond1  = ip & (ev <= 59)
    cond4a = ip & (ev*2 - ang >= 87)    & (ang <= 41) & (ev*2 + ang <= 175) & (ev + ang*1.3 >= 89) & ev.between(59, 72)
    cond4b = ip & (ev + ang*1.3 <= 112) & (ev + ang*1.55 >= 92) & (ev >= 72) & (ev >= 86)
    cond4c = ip & (ang <= 20)           & (ev + ang*2.4 >= 98) & ev.between(86, 95)
    cond4d = ip & (ev - ang >= 76)      & (ev + ang*2.4 >= 98) & (ev >= 95) & (ang <= 30)
    cond3  = ip & (ev + ang*2 >= 116)
    cond2  = ip & (ev + ang*2 <  116)

    sa = pd.Series(0, index=hs25.index)
    sa[cond2]  = 2; sa[cond3] = 3
    sa[cond4a | cond4b | cond4c | cond4d] = 4
    sa[cond5]  = 5; sa[cond6] = 6; sa[cond1] = 1
    hs25["is_icr"] = np.where(sa.isin([4,5,6]), 1, np.where(sa.isin([1,2,3]), 0, np.nan))

    hs25["is_barrel"] = (
        ((es >= 156)    & ang.between(23.5, 32.9))  |
        ((es >= 159.25) & ang.between(21.15, 37.6)) |
        ((es >= 162.5)  & ang.between(21.15, 39.95))|
        ((es >= 165.75) & ang.between(18.8, 47.0))  |
        ((es >= 169)    & (ang >= 18.8))  |
        ((es >= 172.25) & (ang >= 16.45)) |
        ((es >= 175.5)  & (ang >= 14.1))  |
        ((es >= 178.75) & (ang >= 11.75)) |
        ((es >= 182)    & (ang >= 9.4))
    ).fillna(False).astype(int)

    # ── 집계 ─────────────────────────────────────────────────────────────────
    result = (
        hs25.groupby(["BatterId", "Batter", "BatterTeam", "BatterSide"])
        .agg(
            투구수=("PitchCall",    "count"),
            타석=("is_pa",         "sum"),
            인플레이=("is_inplay",  "sum"),
            최고타구속도=("is_exitspeed", "max"),
            평균타구속도=("is_exitspeed", "mean"),
            평균발사각도=("is_angle",     "mean"),
            최고비거리=('is_distance', 'max'),
            하드힛=("is_hardhit",   "sum"),
            icr=("is_icr",         "sum"),
            barrel=("is_barrel",   "sum"),
            스윗스팟=("is_sweetspot",  "sum"),
            땅볼=("is_gb",         "sum"),
            뜬공=("is_fb",         "sum"),
            라인드라이브=("is_ld",  "sum"),
            타구유형=("is_hitype",  "sum"),
            번트=("is_bunt",       "sum"),
        )
        .reset_index()
    )

    result["하드힛%"]      = (result["하드힛"]      / result["인플레이"] * 100).round(1)
    result["정타%"]         = (result["icr"]          / result["인플레이"] * 100).round(1)
    result["배럴%"]      = (result["barrel"]       / result["인플레이"] * 100).round(1)
    result["스윗스팟%"]      = (result["스윗스팟"]       / result["인플레이"] * 100).round(1)
    result["땅볼%"]        = (result["땅볼"]         / result["타구유형"] * 100).round(1)
    result["뜬공%"]        = (result["뜬공"]         / result["타구유형"] * 100).round(1)
    result["라인드라이브%"] = (result["라인드라이브"] / result["타구유형"] * 100).round(1)
    result["번트%"]        = (result["번트"]         / result["타구유형"] * 100).round(1)

    stats = result[["BatterId", "Batter", "BatterTeam", "BatterSide", "투구수", "타석", "인플레이",
                    "최고타구속도", "평균타구속도", "평균발사각도", "최고비거리",
                    "하드힛%", "정타%", "배럴%", "스윗스팟%",
                    "땅볼%", "뜬공%", "라인드라이브%", "번트%"]]
    
    stats = stats.rename(columns={
        "Batter":     "선수명",
        "BatterTeam": "학교"
    })
    
    stats["BatterSide"] = stats["BatterSide"].map({"Right": "우타", "Left": "좌타"})
    stats["BatterSide"] = pd.Categorical(
    stats["BatterSide"], categories=["우타", "좌타"], ordered=True)

    return stats
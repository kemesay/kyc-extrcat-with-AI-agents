"""
Generates synthetic, fictional ID specimens for exercising the universal
engine end-to-end without real personal data:

  passport_uto.png     ICAO 9303 "Utopia" specimen (ERIKSSON ANNA MARIA),
                       with the spec's checksum-valid TD3 MRZ
  eth_style_id.png     bilingual Amharic/English card, dual EC/GC dates
                       (fictional person and number)
  chn_resident_id.png  Chinese resident ID layout with the widely published
                       sample number 11010519491231002X
  td1_id_front.png /   ICAO 9303 Part 5 TD1 identity-card specimen, data on the
  td1_id_back.png      front and the 3x30 MRZ on the back
  td2_permit.png       ICAO 9303 Part 6 TD2 specimen, 2x36 MRZ on the front

Usage: python tests/make_specimens.py [output_dir]   (default tests/fixtures)
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONTS = "/System/Library/Fonts/Supplemental/"
LATIN = FONTS + "Arial.ttf"
LATIN_BOLD = FONTS + "Arial Bold.ttf"
MONO = FONTS + "Courier New Bold.ttf"
ETHIOPIC = FONTS + "KefaIII.ttf"
CJK = FONTS + "Arial Unicode.ttf"


def font(path, size):
    return ImageFont.truetype(path, size)


def card(width=1400, height=900, color=(236, 242, 248)):
    img = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, width - 10, height - 10], outline=(90, 110, 140), width=4)
    draw.rectangle([60, 190, 360, 560], fill=(200, 205, 215), outline=(120, 120, 130), width=2)
    draw.text((150, 360), "PHOTO", font=font(LATIN_BOLD, 34), fill=(140, 140, 150))
    return img, draw


def passport(out: Path):
    img, d = card()
    d.text((420, 40), "UTOPIA", font=font(LATIN_BOLD, 54), fill=(20, 40, 90))
    d.text((420, 110), "PASSPORT / PASSEPORT", font=font(LATIN_BOLD, 36), fill=(20, 40, 90))
    rows = [
        ("Type / Type", "P"), ("Code / Code", "UTO"), ("Passport No. / N° du passeport", "L898902C3"),
        ("Surname / Nom", "ERIKSSON"), ("Given names / Prénoms", "ANNA MARIA"),
        ("Nationality / Nationalité", "UTOPIAN"), ("Date of birth / Date de naissance", "12 AUG 1974"),
        ("Sex / Sexe", "F"), ("Place of birth / Lieu de naissance", "ZENITH"),
        ("Date of issue / Date de délivrance", "16 APR 2007"),
        ("Date of expiry / Date d'expiration", "15 APR 2012"), ("Authority / Autorité", "PASSPORT OFFICE"),
    ]
    y = 190
    for label, value in rows:
        d.text((420, y), label, font=font(LATIN, 20), fill=(90, 90, 110))
        d.text((420, y + 22), value, font=font(LATIN_BOLD, 28), fill=(10, 10, 10))
        y += 54 if y < 600 else 50
        if y > 640:
            break
    d.text((900, 190), "Date of issue", font=font(LATIN, 20), fill=(90, 90, 110))
    d.text((900, 212), "16 APR 2007", font=font(LATIN_BOLD, 28), fill=(10, 10, 10))
    d.text((900, 260), "Date of expiry", font=font(LATIN, 20), fill=(90, 90, 110))
    d.text((900, 282), "15 APR 2012", font=font(LATIN_BOLD, 28), fill=(10, 10, 10))
    d.text((900, 330), "Authority", font=font(LATIN, 20), fill=(90, 90, 110))
    d.text((900, 352), "PASSPORT OFFICE", font=font(LATIN_BOLD, 28), fill=(10, 10, 10))
    d.rectangle([20, 700, 1380, 880], fill=(255, 255, 255))
    mono = font(MONO, 44)
    d.text((40, 715), "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", font=mono, fill=(0, 0, 0))
    d.text((40, 790), "L898902C36UTO7408122F1204159ZE184226B<<<<<10", font=mono, fill=(0, 0, 0))
    img.save(out / "passport_uto.png")


def ethiopia_style(out: Path):
    img, d = card(color=(245, 240, 225))
    d.text((420, 40), "የኢትዮጵያ ብሔራዊ መታወቂያ", font=font(ETHIOPIC, 44), fill=(20, 80, 40))
    d.text((420, 105), "ETHIOPIAN NATIONAL ID CARD (SPECIMEN)", font=font(LATIN_BOLD, 32), fill=(20, 80, 40))
    rows = [
        ("ሙሉ ስም | Full Name", "አበበ ከበደ ተስፋዬ", ETHIOPIC),
        ("", "ABEBE KEBEDE TESFAYE", LATIN_BOLD),
        ("የትውልድ ቀን | Date of Birth", "20/01/1990 | 12/05/1982", LATIN_BOLD),
        ("ጾታ | Sex", "ወንድ | Male", ETHIOPIC),
        ("ዜግነት | Nationality", "ኢትዮጵያዊ | Ethiopian", ETHIOPIC),
        ("የሚያበቃበት ቀን | Date of Expiry", "30/06/2030 | 23/10/2022", LATIN_BOLD),
        ("FAN", "1234 5678 9012 3456", LATIN_BOLD),
    ]
    y = 190
    for label, value, f in rows:
        if label:
            d.text((420, y), label, font=font(ETHIOPIC, 22), fill=(90, 90, 90))
            y += 28
        d.text((420, y), value, font=font(f, 32), fill=(10, 10, 10))
        y += 52
    img.save(out / "eth_style_id.png")


def china_resident(out: Path):
    img, d = card(color=(240, 246, 250))
    d.text((420, 40), "中华人民共和国居民身份证", font=font(CJK, 44), fill=(160, 30, 30))
    d.text((420, 105), "RESIDENT IDENTITY CARD (SPECIMEN)", font=font(LATIN_BOLD, 30), fill=(160, 30, 30))
    rows = [("姓名", "王芳"), ("性别", "女"), ("民族", "汉"), ("出生", "1949年12月31日"),
            ("住址", "北京市朝阳区建国路88号"), ("公民身份号码", "11010519491231002X")]
    y = 200
    for label, value in rows:
        d.text((420, y), label, font=font(CJK, 26), fill=(90, 90, 110))
        d.text((600, y - 4), value, font=font(CJK, 34), fill=(10, 10, 10))
        y += 70
    img.save(out / "chn_resident_id.png")


def _labelled(d, x, y, label, value, size=28):
    d.text((x, y), label, font=font(LATIN, 19), fill=(90, 90, 110))
    d.text((x, y + 22), value, font=font(LATIN_BOLD, size), fill=(10, 10, 10))


def td1_card(out: Path):
    """ICAO 9303 Part 5 specimen: TD1 identity card — data on the front, the
    3-line MRZ on the back (the usual ID-card layout)."""
    img, d = card(color=(232, 240, 236))
    d.text((420, 40), "UTOPIA", font=font(LATIN_BOLD, 50), fill=(20, 70, 50))
    d.text((420, 105), "IDENTITY CARD / CARTE D'IDENTITÉ", font=font(LATIN_BOLD, 30), fill=(20, 70, 50))
    for i, (label, value) in enumerate([("Surname / Nom", "ERIKSSON"), ("Given names / Prénoms", "ANNA MARIA"),
                                        ("Sex / Sexe", "F"), ("Nationality / Nationalité", "UTO"),
                                        ("Date of birth / Date de naissance", "12 08 1974"),
                                        ("Document No. / N° du document", "D23145890"),
                                        ("Date of expiry / Date d'expiration", "15 04 2012")]):
        _labelled(d, 420, 190 + i * 62, label, value)
    img.save(out / "td1_id_front.png")
    back = Image.new("RGB", (1400, 900), (232, 240, 236))
    bd = ImageDraw.Draw(back)
    bd.rectangle([10, 10, 1390, 890], outline=(90, 110, 140), width=4)
    bd.text((60, 60), "Address / Adresse: 12 MAIN STREET, ZENITH", font=font(LATIN, 30), fill=(30, 30, 30))
    bd.rectangle([20, 560, 1380, 880], fill=(255, 255, 255))
    mono = font(MONO, 56)
    for i, line in enumerate(["I<UTOD231458907<<<<<<<<<<<<<<<", "7408122F1204159UTO<<<<<<<<<<<6",
                              "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"]):
        bd.text((70, 585 + i * 95), line, font=mono, fill=(0, 0, 0))
    back.save(out / "td1_id_back.png")


def td2_card(out: Path):
    """ICAO 9303 Part 6 specimen: TD2 card with the 2-line MRZ on the front."""
    img, d = card(width=1500, height=1000, color=(240, 236, 228))
    d.text((420, 40), "UTOPIA", font=font(LATIN_BOLD, 50), fill=(90, 50, 20))
    d.text((420, 105), "RESIDENCE PERMIT / TITRE DE SÉJOUR", font=font(LATIN_BOLD, 30), fill=(90, 50, 20))
    for i, (label, value) in enumerate([("Surname / Nom", "ERIKSSON"), ("Given names / Prénoms", "ANNA MARIA"),
                                        ("Sex / Sexe", "F"), ("Date of birth", "12.08.1974"),
                                        ("Document No.", "D23145890"), ("Valid until", "15.04.2012")]):
        _labelled(d, 420, 180 + i * 58, label, value, 26)
    d.rectangle([20, 760, 1480, 980], fill=(255, 255, 255))
    mono = font(MONO, 50)
    d.text((50, 790), "I<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<", font=mono, fill=(0, 0, 0))
    d.text((50, 880), "D231458907UTO7408122F1204159<<<<<<<6", font=mono, fill=(0, 0, 0))
    img.save(out / "td2_permit.png")


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "fixtures")
    target.mkdir(parents=True, exist_ok=True)
    passport(target)
    ethiopia_style(target)
    china_resident(target)
    td1_card(target)
    td2_card(target)
    print(f"specimens written to {target}")

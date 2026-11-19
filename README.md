# Rakip

Turnuva hazırlığı için: bir rakibin geçmiş oyunlarından **onun gibi oynayan** bir satranç botu üretir,
botun ona ne kadar benzediğini dürüstçe ölçer ve tarayıcıda antrenman maçı yaptırır.

Amaç güçlü oynamak değil, benzer oynamaktır. Rakibin tercih ettiği açılışlara giren, onun yaptığı
hataları yapan bot doğru bottur; Stockfish gibi kusursuz oynayan bir bot bu ürün için başarısızdır.

Her şey ücretsiz altyapıda çalışır:

| İş | Nerede |
|---|---|
| Veri toplama, eğitim, ölçüm | Kaggle Notebooks (yedek: Google Colab) |
| Model ve rapor depolama | Hugging Face Hub |
| Bot hamleleri ve oyun analizi | Kullanıcının tarayıcısı (ONNX Runtime Web, Stockfish wasm) |
| Site | Vercel ya da Cloudflare Pages, statik |

Sunucu ve veritabanı yoktur. Geliştiricinin bilgisayarında eğitim ya da büyük indirme yapılmaz.

## Durum: neyin denendiği, neyin denenmediği

Bunu baştan bilmen gerekir.

**Çalıştırılıp doğrulananlar**

- Veri toplama, temizlik, tempo ayrımı, zamana göre bölme, açılış kitabı, benzerlik skoru, rapor:
  108 birim testi. Lichess, Chess.com ve TWIC'e gerçek isteklerle küçük bir deneme yapıldı.
- Tarayıcıdaki kodlamanın Python ile birebir aynı olduğu (`web/tests/index.html`).
- Tarayıcıda bot: genel Maia-3 modeliyle hamle başına 65-78 ms, model 10,8 MB, önbelleğe alınıyor.
- Tarayıcıda Stockfish, oyun ekranı, oyun sonu geri bildirimi, rapor ve skor sayfaları, mobil görünüm.

**Yazılan ama henüz hiç çalıştırılmayanlar**

- Model (`rakip/model.py`), eğitim döngüsü (`rakip/train.py`), ONNX dışa aktarım (`rakip/export.py`)
  ve uçtan uca hat (`rakip/pipeline.py`). Yerel makinede PyTorch kurulmadığı için bunlar yalnızca
  sözdizimi düzeyinde kontrol edildi.
- Bunları senin bilgisayarın yerine GitHub'ın ücretsiz sunucularında deneyen duman testleri
  hazır (`tests/test_torch_smoke.py`); depoyu GitHub'a gönderdiğinde kendiliğinden çalışır. İçlerinde
  gerçek Maia-3 ağırlıklarının yüklenip anlamlı tahmin ürettiğini doğrulayan bir test de var.
- Sonuç: **ilk Kaggle çalıştırmasından önce GitHub Actions sonucuna bak.** Kırmızıysa eğitim kodunda
  düzeltilecek bir şey vardır.

Sitedeki "Örnek Oyuncu" sayfası (yalnızca `localhost`'ta ya da adrese `?ornek` eklenince görünür)
sentetik veriyle üretilmiştir; gerçek bir ölçüm değildir.

## İlk rakibini eğit: adım adım

### 1. Kodu kendi GitHub hesabına koy

Kaggle, kodu GitHub'dan kurar; bu yüzden depo **herkese açık** olmalı.

```bash
git add -A
```

```bash
git commit -m "Rakip: ilk sürüm"
```

GitHub'da boş bir depo aç (ör. `rakip`), sonra:

```bash
git remote add origin https://github.com/arasarkan/rakip.git
```

```bash
git push -u origin main
```

Depo adresi iki yerde yazılıdır; depoyu başka bir hesaba ya da ada taşırsan ikisini de güncelle:

- `web/config.js` içindeki `githubRepo`
- `tools/build_notebook.py` içindeki `RAKIP_REPO` (sonra `python tools/build_notebook.py` ile not defterini yeniden üret)

Gönderdikten sonra GitHub'da **Actions** sekmesinde iki işin de yeşil olduğunu kontrol et.

### 2. Hesapları aç

- **Kaggle** (kaggle.com): ücretsiz. GPU kullanabilmek için *Settings* sayfasından telefon numaranı doğrula.
- **Hugging Face** (huggingface.co): ücretsiz. Kart bilgisi istenmez.

### 3. Hugging Face token'ı oluştur

Hugging Face'te *Settings > Access Tokens > New token*. Tür: **Write**. Token'ı kopyala; bir daha gösterilmez.

### 4. Not defterini Kaggle'a al

Kaggle'da *Create > New Notebook > File > Import Notebook > GitHub* yolundan deponu seç ve
`notebooks/rakip_train.ipynb` dosyasını içe aktar. (Sitedeki "Yeni rakip eğit" sayfasında, `githubRepo`
ayarını yaptıktan sonra doğrudan bağlantı da çıkar.)

### 5. Token'ı gizli anahtar olarak ekle

Not defterinde *Add-ons > Secrets > Add secret*:

- Label: `HF_TOKEN`
- Value: kopyaladığın token

Ekledikten sonra yanındaki kutuyu işaretleyip not defterine erişim ver. **Token'ı hiçbir hücreye yazma.**
İsteğe bağlı: Lichess indirmelerini hızlandırmak için `LICHESS_TOKEN` da ekleyebilirsin.

### 6. Oturum ayarları

Sağ panelde: **Accelerator: GPU T4 x1**, **Internet: On**.

### 7. Parametreleri doldur

İlk kod hücresinde:

```python
OYUNCU_ADI = "Ad Soyad"          # raporda görünecek ad
TEMPO = "blitz"                   # classical | rapid | blitz | bullet
LICHESS_KULLANICI = "kullaniciadi"
CHESSCOM_KULLANICI = ""
```

Masa başı oyunlar için `TWIC_KULLAN = True`, `OTB_ISIMLER = ["Soyad, Ad"]` ve mümkünse `FIDE_ID` ver.

**Ek tempolar.** Varsayılan olarak komşu tempolardaki oyunlar da eğitime katılır (rapid için klasik ve blitz).
Bunlar yalnızca doğrulama döneminden önce oynanmışsa ve düşük ağırlıkla girer: toplam etkileri hedef tempo
verisinin en çok `EK_VERI_AGIRLIGI` (varsayılan 0,5) katı kadardır. Benzerlik yine yalnızca seçilen tempodaki
oyunlarla ölçülür. Kapatmak için `EK_TEMPOLAR = []`.

**Tempo nasıl belirleniyor?** Süre ayarı varsa ondan. TWIC gibi saat bilgisi olmayan arşivlerde etkinlik
adından: "Blitz", "Armageddon", "Titled Tue" gibi kelimeler blitz; "Rapid" ve bilinen hızlı seriler
(Champions Chess Tour, Global Chess League ...) rapid; geri kalan masa başı turnuvalar klasik. Çevrim içi
oynanmış (Site: Chess.com, Lichess ...) ve temposu anlaşılamayan oyunlar tahmin edilmez, atılır. Not defteri
hangi etkinliği hangi tempoya koyduğunu çıktıda listeler; yanlış görünen bir satır varsa
`rakip/config.py` içindeki listelere eklenmeli.
Kendi PGN dosyaların için onları bir Kaggle veri seti olarak yükle, not defterine ekle ve yolunu
`PGN_YOLLARI` içine yaz.

### 8. Çalıştır

**Save Version > Save & Run All (Commit)**. Böyle çalıştırınca iş bitince oturum kendiliğinden kapanır
ve haftalık GPU kotan boşa harcanmaz. Sekmeyi kapatabilirsin; sonucu *Versions* altında görürsün.

Etkileşimli oturumda (hücreleri elle) çalıştırırsan Kaggle oturumu koddan kapatmaya izin vermez:
iş bitince sağ üstten oturumu kendin durdurmalısın.

Çıktının sonunda şunlar yazar:

- Benzerlik skoru özeti
- `siteye eklemek için depo adı: kullanıcı/rakip-ad-soyad-blitz`

### 9. Siteyi yayınla

Site `web/` klasöründeki düz dosyalardır; derleme adımı yoktur.

- **Vercel:** *Add New > Project*, GitHub deponu seç. *Root Directory:* `web`. *Framework Preset:* Other.
  Build komutu boş kalsın.
- **Cloudflare Pages:** *Create > Pages > Connect to Git*. *Build command:* boş. *Build output directory:* `web`.

### 10. Rakibi siteye bağla

Herkese açık depolar, `rakip-bot` etiketi sayesinde sitede kendiliğinden listelenir. Listede yalnızca
kendi rakiplerini görmek için `web/config.js` içindeki `hubAuthors` alanına Hugging Face kullanıcı adını
yaz. Tek bir depoyu elle eklemek için sitedeki **Rakip ekle** alanını kullan.

Özel (private) depo kullandıysan (`HUB_DEPO_OZEL = True`), sitede "Özel depo kullanıyorum" bölümüne
salt-okunur bir token girmen gerekir. Token yalnızca o tarayıcıda saklanır.

### Seans koparsa

Aynı parametrelerle yeniden çalıştır. Her epoch sonunda checkpoint Hub'a yüklenir; eğitim son
checkpoint'ten devam eder. Veri ya da ayarlar değiştiyse eski checkpoint kullanılmaz, baştan başlanır.

### Colab'da çalıştırmak (yedek)

Aynı not defteri Colab'da da çalışır; ortamı kendisi algılar. Token'ı soldaki anahtar simgesinden
(`HF_TOKEN`) ekle, *Runtime > Change runtime type > T4 GPU* seç. İş bitince çalışma zamanı kendiliğinden
serbest bırakılır.

## Süre ve kota

Kaggle haftada yaklaşık 30 saat GPU verir ve bir seans en çok 12 saat sürer. Tahminler (T4, ölçülmedi):

| Oyun sayısı | Toplam süre |
|---|---|
| ~300 | 15-20 dk |
| ~1000 | 30-40 dk |
| ~10 000 | 1,5-2 saat |

Eğitim başlamadan önce not defteri gerçek hızı ölçüp tahmini süreyi yazdırır. Tahmin 10 saati aşarsa
önce epoch sayısını, sonra veri miktarını (en yeni pozisyonları tutarak) kendisi düşürür. Sürenin çoğu
GPU'da değil, CPU'daki Stockfish değerlendirmesinde geçer.

## Benzerlik skoru nasıl hesaplanıyor?

Kaynak: `rakip/similarity.py` (formül dosyanın başında) ve sitedeki "Bu skor nasıl hesaplanıyor?" sayfası.

**Ölçüm**

1. Rakibin oyunlarının **en yeni %15-20'si** eğitimden önce ayrılır (test seti). Bot bunları eğitimde de
   açılış kitabında da görmez. `tests/test_split.py` bunu sınar: aynı oyunun başka kaynaktan gelen
   kopyası ya da test döneminden yeni bir eğitim oyunu hata fırlatır.
2. Test oyunlarındaki her pozisyonda botun ilk tahmini, rakibin gerçekten oynadığı hamleyle karşılaştırılır.
3. Skoru şişiren hamleler sayılmaz: **açılış kitabındaki pozisyonlar** ve **tek yasal hamlesi olan pozisyonlar**.
4. İki referans hesaplanır:
   - **Genel model:** kimseye özel eğitilmemiş Maia-3. Reytingi, rakibi en iyi tahmin ettiği değere
     doğrulama setinde ayarlanır (yani taban olabildiğince güçlüdür).
   - **Stockfish'in en iyi hamlesi.**

**Formül**

```
Ana skor = 100 x ( 0,6 x min(1, m / T)  +  0,2 x Açılış  +  0,2 x Hata )
```

- `m`: botun rakiple aynı hamleyi oynama oranı.
- `T`: ulaşılabilir tavan. En az 0,75; rakibin açılış kitabı dışında tekrar eden pozisyonlardaki kendi kendisiyle tutarlılığı
  daha yüksekse o kullanılır. 0,75 bir varsayımdır; veri onu yalnızca yükseltebilir, yani skoru yalnızca
  sıkılaştırabilir.
- `Açılış`: rakibin yeni oyunlarının ilk 8 hamlesindeki seçimlerinin ne kadarı botun kitabında var.
- `Hata`: yarısı hata sıklığının benzerliği (açılış / orta oyun / oyun sonu ayrı ayrı), yarısı rakibin hata
  yaptığı pozisyonlarda botun da hata yapma oranı. Hata, Stockfish'e göre kazanma şansını en az 10 puan
  düşüren hamledir. Botun burada değerlendirilen hamlesi ilk tahmini değil, gerçekte oynayacağı gibi
  olasılık dağılımından örneklenen hamledir.

Hata profili hesaplanamazsa ağırlıklar 0,75 / 0,25 olur.

**Hamle çeşitliliği (sıcaklık).** Bot hamlesini olasılıklara göre seçer; sıcaklık bu seçimin ne kadar dağınık
olacağını belirler. Sıcaklık 1'de bot oyuncudan çok daha sık hata yapıyordu (Eric Rosen: oyuncu %6, bot %10).
Artık doğrulama setinde (test setine dokunmadan) modelin en olası 6 adayı Stockfish ile değerlendirilir ve botun
beklenen ciddi hata sıklığı oyuncununkine en yakın olan sıcaklık seçilir (`rakip/calibrate.py`). Değerlendirilemeyen
düşük olasılıklı hamleler temkinli biçimde hata sayılır. Genel model de aynı yöntemle ayrı ayarlanır. Seçilen değer
sitede botun varsayılan ayarıdır ve hata profili bu sıcaklıkla ölçülür; hamle uyumu (ilk tahmin) etkilenmez.

**Dürüstlük kuralları**

- Aynı formül genel model için de hesaplanır ve yan yana gösterilir.
- Bot genel modelden (ya da Stockfish'ten) istatistiksel olarak anlamlı şekilde iyi değilse skor gri ve
  "belirsiz" gösterilir, nedeni cümleyle yazılır. Anlamlılık, oyunları yeniden örnekleyen bir güven
  aralığıyla ölçülür; aynı oyunun hamleleri bağımsız sayılmaz.
- 200'den az oyun: güven Düşük ve skor gri. 200-1000: Orta. 1000'den fazla: Yüksek. Ölçüm 1000'den az
  pozisyona dayanıyorsa bir kademe düşer.
- Yayınlanan skor, yayınlanan dosyanın (küçültülmüş ONNX modelinin) skorudur. int8 sürümü yalnızca hamle
  eşleşmesini 0,5 puandan fazla düşürmüyorsa kullanılır; karşılaştırma tablosu `scores.json` içindedir.
- Skorlar hep aşağı yuvarlanır.

**Gerçekçi beklenti:** insanlar aynı pozisyonda bile her zaman aynı hamleyi oynamaz. En iyi kişisel
modeller bile 10 hamlenin 6-7'sinden fazlasını tutturamaz. 100 puan "her hamlesi aynı" değil,
"ulaşılabilir tavana ulaştı" demektir.

## Sistemin sınırları

- **Bot hesap yapmaz.** Yalnızca "bu oyuncu burada ne oynardı" sorusunu yanıtlar; önünü aramaz. Uzun
  zorunlu varyantlarda gerçek oyuncudan zayıf kalabilir.
- **Güçlü oyuncular ve masa başı oyunlar.** Temel model (Maia3-5M) çevrim içi Lichess oyunlarıyla eğitildi.
  Büyükusta düzeyinde ve klasik masa başı oyunlarda benzerlik daha düşük çıkabilir. FIDE reytingleri
  Lichess ölçeğine kaba bir kaydırmayla (+200) çevrilir.
- **Az veri.** 200 oyunun altında yalnızca oyuncu vektörü ve hamle başlığı eğitilir; kişiselleştirme sınırlı kalır.
- **Çok güçlü oyuncularda kişiselleştirme zor.** İlk gerçek deneme (Magnus Carlsen, rapid, 720 eğitim oyunu)
  genel modelden farklı bir bot üretemedi: ince ayar doğrulama kaybını iyileştirmedi ve Stockfish, Carlsen'in
  hamlelerini her iki modelden de iyi tahmin etti (%56'ya karşı %54). Skor bu yüzden "belirsiz" gösterildi.
  Öğrenme oranları düşürüldü ve ek tempo verisi eklendi; etkisi henüz ölçülmedi.
- **Masa başı oyunlarda saat yok.** TWIC ve çoğu PGN arşivi saat bilgisi içermez; zaman kullanımı bölümü boş kalır.
- **İsim eşleştirme.** TWIC isimleri "Soyad,A" biçimindedir ve çakışabilir. `FIDE_ID` vermek güvenlidir.
- **Repertuvar değişimi.** Rakip test döneminde yeni açılışlara geçtiyse skor düşer; bu gerçeğin kendisidir.
- **Tarz ve zayıf nokta etiketleri kabadır.** "Taktik / pozisyonel" göstergesi oyun yapısından çıkan
  sayımlara dayanır; popülasyonla karşılaştırma içermez.
- **Etkileşimli Kaggle oturumu koddan kapatılamaz.** Kendiliğinden kapanma yalnızca "Save & Run All" ile garantidir.
- **Site CDN'lere bağımlıdır.** ONNX Runtime, chess.js, chessground ve Stockfish jsDelivr/unpkg üzerinden
  yüklenir. Demo modeli üçüncü bir Hugging Face deposundan gelir (`web/config.js` içinde değiştirilebilir).
- **Tavan bir varsayımdır.** 0,75 değeri ölçülmüş bir sabit değildir.

## Depo yapısı

```
rakip/                Python paketi (Kaggle/Colab'da kurulur)
  sources/            Lichess, Chess.com, TWIC, PGN dosyaları
  http.py             hız sınırı, 429 geri çekilme, önbellek
  games.py            oyun kaydı, temizlik, oyuncu eşleştirme
  split.py            zamana göre bölme + sızıntı kontrolleri
  book.py             açılış kitabı
  encoding.py         Maia-3 girdi/çıktı kodlaması
  features.py         model örnekleri
  model.py            Maia-3 mimarisi + oyuncu vektörü          (torch)
  train.py            ince ayar, checkpoint, devam              (torch)
  engine.py           Stockfish değerlendirmesi
  similarity.py       benzerlik skoru
  report.py           rakip raporu
  export.py           ONNX + quantize                           (torch)
  pipeline.py         uçtan uca hat                             (torch)
notebooks/rakip_train.ipynb   parametreli Kaggle/Colab not defteri
tools/                not defteri, test örnekleri ve örnek veri üreticileri
tests/                birim testleri
web/                  statik site (derleme yok)
```

Her rakibin Hub deposunda şunlar bulunur: `model.onnx`, `meta.json`, `scores.json`, `report.json`,
`book.json`, `split.json` ve eğitim checkpoint'leri.

## Geliştirme

Hafif testler (torch gerektirmez, birkaç saniye):

```bash
pip install -e ".[test]"
```

```bash
pytest -q
```

Siteyi yerelde açmak için `web/` klasörünü herhangi bir statik sunucuyla sun:

```bash
python -m http.server 5173 --directory web
```

Tarayıcı testleri: `http://localhost:5173/tests/index.html`.

## Lisans ve teşekkür

Bu proje **AGPL-3.0** ile lisanslıdır; siteyi yayınlarsan kaynak kodunu da açık tutman gerekir.

- Temel model: [Maia-3](https://github.com/CSSLab/maia3) (CSSLab, Toronto Üniversitesi; AGPL-3.0).
  Monroe ve ark., *Chessformer: A Unified Architecture for Chess Modeling*, ICLR 2026.
- Analiz: [Stockfish](https://stockfishchess.org) ve [stockfish.js](https://github.com/nmrugg/stockfish.js) (GPL-3.0).
- Tahta: [chessground](https://github.com/lichess-org/chessground) (GPL-3.0). Kurallar: [chess.js](https://github.com/jhlywa/chess.js).
- Veri: Lichess API, Chess.com Published-Data API, The Week in Chess.

# 11 — Tiêu chí nghiệm thu đề xuất

Các threshold dưới đây là **đề xuất để duyệt trước benchmark**, không kết quả. Điều kiện600 s và profile720p là yêu cầu đã ban hành; quality thresholds chi tiết cần khóa với corpus trước phát triển.

## Corpus và oracle

Tối thiểu6video thật600 s720p cho strata ítchữ/bảngdày/sự kiện ngắns/motion/typography/art/systemupdates;1080p vàVFR thêm riêng. Không chỉ chọn clip dễ. Có>=100events0,3–0,5s phân bố vị trí/phase trên corpus và>=100sự kiện critical; số lượng phải báo denominator từng nhóm. HaiannotatorsTrung–Việt gáninterval/polygon/text/type/importance/translation, adjudication. Dev/test khácseries, không tune lên holdout. Corpuschưa có thì quality=UNKNOWN.

| Gate | Threshold đề xuất | Cách đo |
|---|---|---|
| Performance | Mọi acceptance runcold/warm<=600 s, mục tiêuheadroom<=520s | Request→verifiedoutput, gồmload / network / QC đầy đủ; reportmọilần |
| Candidate/detection | Criticalcandidate recall>=99%; recall sự kiện tường thuật>=95% | Stagewisegold, không đếmframe thayevents |
| Sự kiện ngắn | Recall>=95% riêng0,3–0,5s; khôngmisscriticalgánnhãn | nhiềuphase/position, khônggộpvàoclipdài |
| Localizedcritical | **0sự kiện critical bị bỏ** trênholdout | OCR→importance→translate→render toànđường; reviewchưa giải quyết=chưađạt |
| OCR | CER<=5%trênselectedreadabletext; name/skilllabelaccuracy>=99% | Giữraw/normalized, báo stratumart/smallriêng |
| Numbers | 100% số/unit/range/negative/key-value đúng trêncritical | Typedround-trip+humanoracle |
| Translation | 0lỗi critical về nghĩa / tên / phủ định; >=95%events đápứngrubric | Hai ngườiđánhgiá, consistencyglossary100% approvedterms |
| Timing | P95boundaryerror<=100ms; max<=300ms choeventthường | MatchgoldPTS. Sự kiện ngắnmax<=min(100ms,duration/4) đểkhôngtrôi hết event |
| Eventassociation | Khôngmergechồngrevisionstat, cut hoặc2vịtrí; identityerrors<=1% | ID/revisiongold, repeatedappearancetest |
| Spatial/tracking | Level1anchor nằm đúng region; trackedpolygon IoU>=0,7 và jitterP95<=3px720p nếuLevel3enabled | Frame-timeannotations, báoocclusionunresolved |
| Layout | 0va chạm nghiêm trọng; không tràn khung / không lỗi glyph; đọcđược | Safezone/maskchecks+reviewframe; minfont20px720p làpolicy thử, cầnreviewcontrast |
| Media | Fulloutputdecode không lỗi, duration/PTS đúng, playable | ffprobe+fulldecode; CFRframecount/lastPTSwithin1frame, VFRPTSsequencecontract |
| Audio | Stream copy payload/decodedcontent giữ nguyên, drift<=1frame nguồn | Stream timestamps/hashpacketpayloads nếu copy; decodedPCMcheck nếucontainer/headerskhác |
| Resource | KhôngOOM, không chuyển provider âm thầm, VRAMprocess<=3,2GiB/RSS<=8GiBproposed | Time-sampledWDDM/RSS+ORTnode trace; xácnhậnfreeRAM vàshared/displaybudget |
| Compatibility | ModeAcriticaloverlays không mất/che sauSubAI ởconfigđược yêu cầu | Matrix[08](08_EXTERNAL_TOOL_COMPATIBILITY.md), version/hashlocked |
| Reliability | Resume/cancel/crash khôngcorrupt/sourceoverwrite | Provenance/checkpoint/hash; reportnevermarkpartialcomplete |

## Định nghĩa metric

Candidate recall: gold event có ít nhất một proposal hợp lệ tronginterval. Event recall match1:1 theo temporal IoU>=0,5 + polygon không gian + text normalized criteria; propernames/numerals vẫn exact khicritical. Localized recall chỉ tính có overlay đúngnghĩa/geometry/time/legibility ởvideo, không chỉ JSON. Stageconfusion/loss ledger tránh detectiongoodnhưngsemanticsbỏsót.

CER tính Levenshteincharacters; normalization không xóa số/ký hiệu để điểm đẹp. Boundaryerror lấy tuyệtđối start/end riêng, uncertaintyreported. Microaverages không che stratumfail; lỗi criticalfail ngay dù overallrecallcao. 100sự kiện critical không chứng minh recall phổquát; confidenceinterval/binomial bounds vàsamplelimits phải ghi.

## Thời gian và quality độc lập

Chạy3cold+5warm/profile, sustained>=30 phút, report min/median/max/P95 vàfailures. PASS chỉ khi media/quality/performance/compatibility/license đềuPASS. NEEDS_REVIEW giữ thấy nhưng chưa Việt hóa và khônghumanfixngoàiđồnghồđểclaimautomation. Outputplayablepartial vẫn usefulartifact, nhưng không nghiệmthu.

Không sinhsyntheticđểthayrealcorpus; synthetic chỉ stress timeline/color/fontedgecases. Khôngbenchmark SubAIstage đểthayVNLE. 1080p không đạt thìbáo riêng, khôngdownscaleâm thầm. Gatefailing cần evidence bottleneck rồi duyệt sửa, khôngđổi model/config giữa run.

## Gate vùng loại trừ do người dùng

100% pixels text trong active ROI không vào detector/recognizer; spy model input/candidate ledger, không chỉ output thiếu sub. Test nhiều rect/polygon, intervals, rotation/SAR/resize và biên. Report EXCLUDED_BY_USER và coverage; denominator quality tính phần được yêu cầu xử lý, kèm toàn bộ phạm vi loại trừ để không giấu coverage. Không tự chọn/nới ROI. Overlay mặc định không chồng ROI dành SubAI.

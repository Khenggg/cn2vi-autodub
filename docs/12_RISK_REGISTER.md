# 12 — Sổ rủi ro

Ngày09/10/2026. Likelihood/impact là nhận định thiết kế, chưa đo; owner là vai trò tương lai, không nhân sự đã phân công.

| ID | Rủi ro / khả năng–ảnh hưởng | Evidence/trigger | Giảm thiểu và owner / exit |
|---|---|---|---|
| R01 | Detectorcoverage4Hz quáđắt, cao–critical | ~2400calls trướctriggers; cap70s chưa đo | CVowner: detector/proposalbenchmark+sổ ghi nhận mất sự kiện; không bỏ sự kiện; FAILSLA nếuvượt |
| R02 | Miss0,3s/chữnhỏ, cao–critical | Cadencethưa/coarsefilters | CVowner: watchdog đầy đủ+localchanges+gold theo phase; passrecall sự kiện ngắncriticalzeroomit |
| R03 | Densepanels/cập nhật chỉ số bịmerge, cao–critical | Fuzzytextmatching/cachepanel | Eventowner: typedrevision/valuecropinvalidations; số liệu 100% |
| R04 | CPU ASS/render/QC nghẽn, cao–high | FFmpegsourcefilterCPU,90fpsrenderbudget | Mediaowner: baselinefloor+ABroutes native / GPU; QC đầy đủcostkhông giấu |
| R05 | GPUfallback/driver DLLs, vừa–high | ORTproviderconfignotnode trace | Runtimeowner: cleanenvpinmatrix+profiling; không fallback ẩn |
| R06 | VRAM 4 GB/RAM/thermal, cao–high | Multiplemodels / frame surfaces / laptop | Runtimeowner: một worker GPU/bytequeues, sustainedtest; noOOM/throttleSLAfail |
| R07 | SubAIxóa/đè/cropnhãn, cao–critical | ModeAinputoverlay maylooklikesub | Integrationowner: realmatrix[08]; nosửa SubAI; Bchỉsauapproval |
| R08 | ModeBmapping khôngđơnđiệu, vừa–critical | Cut/reorder/speed/resize | Integrationowner: anchor+piecewisemappingconfidence; failclosedreview |
| R09 | Saiimportance/tên/nghĩa, cao–critical | Rules/OCRlackcontext | Languageowner: glossary+eventcontext+uncertainty; noLLMselfgradeoracle |
| R10 | API timeout/cost/privatecontent, vừa–high | Providerchưa chốt | Languageowner: adapter/noimplicitAPI, boundedrequests, unknowncostreported; quality FAILnotfallback |
| R11 | Copyleft/weights unknown, cao–high | MangaGPL/AGPL,CoTrackerNC,TransDETRdeclaration | Licenseowner: source+weights/fonts/runtimeBOM; nocopyunverified; blockdistributionuntilaudit |
| R12 | Color/alpha/chromacorruption, vừa–high | GPUrouteformatconversion | Mediaowner: color/glyph/alphaoracle+decodedchecks, keepmetadata |
| R13 | PTS/VFR/Bframes/rotation sai, vừa–critical | index/fpsorfloatrounding | Mediaowner: rationalPTS/sourcegeometrycontracts; QC theo frame |
| R14 | OCRtextpromptinjection/hallucinations, vừa–high | Videoarbitrarytextinsemanticrequest | Languageowner: treatasdata,no toolprivileges, schema / kiểm số |
| R15 | Corpusoverfit/benchmark chỉ lấy lần tốt nhất, cao–high | Mộtvideodễvàmetricsoverall | QAowner: seriesholdout/strata/repeatedruns/mẫu số critical |
| R16 | Chữdịchquádài0,3s, cao–high | OCRđúngnhưngkhông đọc kịp | Productowner: explicitcondensationpolicy/annotation+review; notextendtimeâm thầm |
| R17 | Packaging/envleak/longpaths, vừa–high | CopiesSubAIRuntime/autoDownloads | Runtimeowner: owncleanenv/folderdist/fontmodel hash/license; test đường dẫn Unicode nguồn |
| R18 | Mấtlegacy/secretleakreset, thấp–critical saubackup | History/dirtyignoreddata | Primary: verifiedcopy+originals+bundleignored; unreadablecachepreserved; noGitpushsecrets |
| R19 | Agentreportclaimkhôngevidence, vừa–high | UnsupportedFPS/weightslicense/generalizedlegalclaims | Primarysourceverify, discardunmeasurednumbers; noautoretryfailedreport |
| R20 | Metadata720p khôngchặnmật độ chữ, cao–high | Infinitecontent/profileunspecified | Product+QA: agreedrealcorpus+densitycounters; không ẩninputexclusion |

## Câu hỏi chưa giải quyết

Corpus/quyềnmedia/nhãncritical; exactCPU/RAM/driver/power máynghiệmthu; approvedseriesglossary; translatorlocal/API vàprivacy/cost; Level1annotationso với thaychữ; densitydistributionvàrelevanceoracle; SubAI version/settings vàtimestamp/resizebehavior; distribution license/businesscontext; chính sách đọc sự kiện ngắn; 1080p làSLAhaybest-effort; mappingBchoeditphi tuyến.

Các câu hỏi này chưa ngăn nghiên cứu, nhưng ngăn khẳng định product/SLAready. Thu thậptạigatesP0–P2, không tự giảđịnhđãđượcchốt từCN2VI. Không tự bổ sung cloud/backend/UI trong scope hiện tại.

## R21 — Vùng loại trừ sai hoặc mapping lệch

ROI quá rộng có thể loại chữ quan trọng. Preview source geometry/time, user-confirmed ROI/hash, test biên và report coverage. Không tự nhận diện lại hoặc nới vùng. User exclusion là scope override đã chốt, không là thuật toán tự loại chữ.

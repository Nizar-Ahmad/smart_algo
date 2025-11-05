using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Web.UI;

namespace WebApp
{
    public partial class Default : System.Web.UI.Page
    {
        private class ScoredMask
        {
            public bool[] m;
            public double fitness;
        }

        // Session State 
        private List<string> Headers
        {
            get => (List<string>)(Session["Headers"] ?? new List<string>());
            set => Session["Headers"] = value;
        }
        private double[][] X // features: rows x cols
        {
            get => (double[][])Session["X"];
            set => Session["X"] = value;
        }
        private string[] yRaw // original target values as string
        {
            get => (string[])Session["yRaw"];
            set => Session["yRaw"] = value;
        }
        private double[] y // numeric target (for regression or encoded classes 0..C-1)
        {
            get => (double[])Session["y"];
            set => Session["y"] = value;
        }
        private bool IsClassification
        {
            get => (bool)(Session["IsCls"] ?? true);
            set => Session["IsCls"] = value;
        }
        private Dictionary<string, double> ClassMap
        {
            get => (Dictionary<string, double>)(Session["ClassMap"] ?? new Dictionary<string, double>());
            set => Session["ClassMap"] = value;
        }

        private static readonly Random rnd = new Random();

        // Page
        protected void Page_Load(object sender, EventArgs e)
        {
            if (!IsPostBack)
            {
                // Empty initial UI
                litResult.Text = "";
                litLoadInfo.Text = "";
                litHistory.Text = "<div class='text-muted'>لا يوجد سجل بعد.</div>";
                litCharts.Text = "";
            }
        }

        // CSV Loading 
        protected void btnLoadCsv_Click(object sender, EventArgs e)
        {
            if (!fuCsv.HasFile)
            {
                litLoadInfo.Text = "<div class='text-danger'>الرجاء اختيار ملف CSV.</div>";
                return;
            }

            try
            {
                using (var sr = new System.IO.StreamReader(fuCsv.FileContent, Encoding.UTF8, true))
                {
                    var lines = new List<string>();
                    while (!sr.EndOfStream)
                    {
                        var line = sr.ReadLine();
                        if (!string.IsNullOrWhiteSpace(line)) lines.Add(line);
                    }
                    LoadFromLines(lines);
                }
                litLoadInfo.Text = "<div class='text-success'>تم تحميل CSV بنجاح.</div>";
            }
            catch (Exception ex)
            {
                litLoadInfo.Text = $"<div class='text-danger'>فشل التحميل: {Server.HtmlEncode(ex.Message)}</div>";
            }
        }

        protected void btnUseSample_Click(object sender, EventArgs e)
        {
            // صنع عينة بسيطة (تصنيف ثنائي) 200x20 + target
            int n = 200, d = 20;
            var headers = new List<string>();
            for (int j = 0; j < d; j++) headers.Add("f" + (j + 1));
            headers.Add("target");

            var lines = new List<string> { string.Join(",", headers) };
            for (int i = 0; i < n; i++)
            {
                var row = new List<string>();
                // صنع features
                for (int j = 0; j < d; j++)
                {
                    double val = NextGaussian(0, 1);
                    if (j < 4) val += (i % 2 == 0 ? 1.5 : -1.5); // أول 4 ميزات مؤثرة
                    row.Add(val.ToString(CultureInfo.InvariantCulture));
                }
                string cls = (i % 2 == 0) ? "A" : "B";
                row.Add(cls);
                lines.Add(string.Join(",", row));
            }
            LoadFromLines(lines);
            litLoadInfo.Text = "<div class='text-success'>تم تحميل بيانات عينة.</div>";
        }

        private void LoadFromLines(List<string> lines)
        {
            if (lines.Count == 0) throw new Exception("CSV فارغ.");
            var headParts = SplitCsvLine(lines[0]);
            bool hasHeader = headParts.All(p => !IsNumeric(p));
            List<string> headers;
            int start = 0;

            if (hasHeader)
            {
                headers = headParts;
                start = 1;
            }
            else
            {
                int cols = SplitCsvLine(lines[0]).Count;
                headers = Enumerable.Range(0, cols).Select(i => "C" + (i + 1)).ToList();
            }

            // نفترض أن العمود الأخير Target مبدئياً 
            Headers = headers;

            // ملأ Dropdown بأسماء الأعمدة
            ddlTarget.Items.Clear();
            foreach (var h in Headers) ddlTarget.Items.Add(h);
            ddlTarget.SelectedIndex = Headers.Count - 1; // آخر عمود target غالباً

            // KPIs UI reset
            PushKpi(samples: "—", feats: "—", bestScore: "—", gaCount: "—");
            litCharts.Text = "";
            litResult.Text = "";
        }

        // نحفظ raw lines في الجلسة عند التحميل
        private void LoadFromLinesAndStore(List<string> lines)
        {
            Session["RawLines"] = lines;
            LoadFromLines(lines);
        }

        // نعدّل دوال التحميل لتخزين الأسطر
        protected override void OnInit(EventArgs e)
        {
            base.OnInit(e);
            btnLoadCsv.Click += (s, ev) =>
            {
                if (fuCsv.HasFile)
                {
                    using (var sr = new System.IO.StreamReader(fuCsv.FileContent, Encoding.UTF8, true))
                    {
                        var lines = new List<string>();
                        while (!sr.EndOfStream)
                        {
                            var line = sr.ReadLine();
                            if (!string.IsNullOrWhiteSpace(line)) lines.Add(line);
                        }
                        Session["RawLines"] = lines;
                        LoadFromLines(lines);
                        litLoadInfo.Text = "<div class='text-success'>تم تحميل CSV بنجاح.</div>";
                    }
                }
                else
                {
                    litLoadInfo.Text = "<div class='text-danger'>الرجاء اختيار ملف CSV.</div>";
                }
            };
            btnUseSample.Click += (s, ev) =>
            {
                // نفس العينة أعلاه
                int n = 200, d = 20;
                var headers = new List<string>();
                for (int j = 0; j < d; j++) headers.Add("f" + (j + 1));
                headers.Add("target");
                var lines = new List<string> { string.Join(",", headers) };
                for (int i = 0; i < n; i++)
                {
                    var row = new List<string>();
                    for (int j = 0; j < d; j++)
                    {
                        double val = NextGaussian(0, 1);
                        if (j < 4) val += (i % 2 == 0 ? 1.5 : -1.5);
                        row.Add(val.ToString(CultureInfo.InvariantCulture));
                    }
                    string cls = (i % 2 == 0) ? "A" : "B";
                    row.Add(cls);
                    lines.Add(string.Join(",", row));
                }
                Session["RawLines"] = lines;
                LoadFromLines(lines);
                litLoadInfo.Text = "<div class='text-success'>تم تحميل بيانات عينة.</div>";
            };
        }

        // RUN 
        protected void btnRun_Click(object sender, EventArgs e)
        {
            if (Headers.Count == 0)
            {
                litResult.Text = "<div class='alert alert-danger'>الرجاء تحميل البيانات أولاً.</div>";
                return;
            }

            string targetName = ddlTarget.SelectedValue;
            int tIdx = Headers.FindIndex(h => h == targetName);
            if (tIdx < 0)
            {
                litResult.Text = "<div class='alert alert-danger'>لا يمكن العثور على عمود الهدف.</div>";
                return;
            }

            var lines = (List<string>)Session["RawLines"];
            if (lines == null)
            {
                litResult.Text = "<div class='alert alert-danger'>يرجى إعادة تحميل الملف (انتهت الجلسة المؤقتة).</div>";
                return;
            }

            // بناء مصفوفات
            BuildXYFromLines(lines, tIdx);

            // تحديد نوع المشكلة
            string forced = ddlProblemType.SelectedValue;
            if (forced == "cls" || forced == "reg")
                IsClassification = forced == "cls";
            else
                InferProblemType();

            // قراءة بارامترات
            int population = ParseInt(txtPop.Text, 30, 5, 500);
            int generations = ParseInt(txtGen.Text, 25, 5, 1000);
            int kfolds = ParseInt(txtKFolds.Text, 3, 2, 10);
            int knnK = ParseInt(txtKnnK.Text, 5, 1, 50);
            double lambda = ParseDouble(txtPenalty.Text, 0.02, 0.0, 1.0);
            int topK = ParseInt(txtTopK.Text, 30, 1, Math.Max(1, Headers.Count - 1));

            // أعمدة الميزات (كل ما عدا target)
            var featIdxs = Enumerable.Range(0, Headers.Count)
                                     .Where(i => i != tIdx).ToArray();

            // قياس الأداء للمقارنات
            var swAll = Stopwatch.StartNew();
            double scoreAll = EvaluateSubset(featIdxs.ToArray(), kfolds, knnK);
            swAll.Stop();

            var topKIdxs = UnivariateTopK(featIdxs, topK);
            var swTop = Stopwatch.StartNew();
            double scoreTop = EvaluateSubset(topKIdxs, kfolds, knnK);
            swTop.Stop();

            // GA
            int D = featIdxs.Length;
            var gaRes = RunGA(D, population, generations, kfolds, knnK, lambda, featIdxs);

            // عرض النتائج
            RenderResultTables(featIdxs, scoreAll, swAll.ElapsedMilliseconds,
                               topKIdxs, scoreTop, swTop.ElapsedMilliseconds,
                               gaRes, tIdx);

            // تشارتات
            RenderCharts(scoreAll, scoreTop, gaRes.Score,
                         gaRes.SelectedFeatureIdxs.Select(i => Headers[i]).ToArray());

            // KPIs
            PushKpi(X.Length.ToString(), D.ToString(),
                    gaRes.Score.ToString("0.000"), gaRes.SelectedFeatureIdxs.Length.ToString());

            // سجل
            AppendHistory(IsClassification ? "Classification" : "Regression",
                          scoreAll, scoreTop, gaRes.Score, gaRes.SelectedFeatureIdxs.Length);
        }

        //Build X / y
        private void BuildXYFromLines(List<string> lines, int targetCol)
        {
            var headParts = SplitCsvLine(lines[0]);
            bool hasHeader = headParts.All(p => !IsNumeric(p));
            int start = hasHeader ? 1 : 0;
            var rows = new List<string[]>();
            for (int i = start; i < lines.Count; i++)
                rows.Add(SplitCsvLine(lines[i]).ToArray());

            int cols = rows[0].Length;
            int n = rows.Count;

            // تنظيف صفوف ناقصة الطول
            rows = rows.Where(r => r.Length == cols).ToList();
            n = rows.Count;

            // بناء X_raw (string) و target raw
            var featIdxs = Enumerable.Range(0, cols).Where(i => i != targetCol).ToArray();
            int d = featIdxs.Length;

            var Xnum = new double[n][];
            var yraw = new string[n];

            for (int i = 0; i < n; i++)
            {
                Xnum[i] = new double[d];
                // target raw as string
                yraw[i] = rows[i][targetCol];

                for (int j = 0; j < d; j++)
                {
                    string s = rows[i][featIdxs[j]];
                    if (!TryParseDouble(s, out double v))
                    {
                        // حاول تحويل فئات إلى رقمية (one-hot غير ممكن هنا) => نعطي 0 إن لم يمكن
                        v = 0;
                    }
                    Xnum[i][j] = v;
                }
            }

            // قياسي: نطبع ميزات (z-score)
            X = Standardize(Xnum, out var means, out var stds);

            // y: classification -> encode classes إلى [0..C-1]، regression -> double
            if (ddlProblemType.SelectedValue == "reg")
            {
                IsClassification = false;
                y = yraw.Select(s => ParseDoubleSafe(s)).ToArray();
                yRaw = yraw;
                ClassMap = new Dictionary<string, double>();
            }
            else if (ddlProblemType.SelectedValue == "cls")
            {
                IsClassification = true;
                var map = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
                int next = 0;
                y = new double[n];
                for (int i = 0; i < n; i++)
                {
                    string key = yraw[i];
                    if (!map.ContainsKey(key)) map[key] = next++;
                    y[i] = map[key];
                }
                ClassMap = map;
                yRaw = yraw;
            }
            else
            {
                // auto infer
                var (isCls, map, ynum) = InferY(yraw);
                IsClassification = isCls;
                ClassMap = map ?? new Dictionary<string, double>();
                y = ynum;
                yRaw = yraw;
            }

            // تحديث KPIs الأساسية
            PushKpi(X.Length.ToString(), X[0].Length.ToString(), "—", "—");
        }

        private void InferProblemType()
        {
            // إذا كانت y أخذت عددًا قليلاً من القيم الفريدة أو ليست أرقامًا => تصنيف
            var distinct = yRaw.Distinct(StringComparer.OrdinalIgnoreCase).Take(50).Count();
            bool numericAll = yRaw.All(s => TryParseDouble(s, out _));
            IsClassification = !(numericAll && distinct > Math.Min(15, yRaw.Length / 10));
        }

        private (bool isCls, Dictionary<string, double> map, double[] ynum) InferY(string[] raw)
        {
            bool allNum = raw.All(s => TryParseDouble(s, out _));
            if (allNum)
            {
                var vals = raw.Select(ParseDoubleSafe).ToArray();
                // إذا القيم الفريدة قليلةف
                int distinct = vals.Distinct().Take(50).Count();
                if (distinct <= 10) // threshold
                {
                    var classes = vals.Select(v => v.ToString(CultureInfo.InvariantCulture)).ToArray();
                    var map = new Dictionary<string, double>();
                    int next = 0;
                    double[] enc = new double[vals.Length];
                    for (int i = 0; i < vals.Length; i++)
                    {
                        string key = classes[i];
                        if (!map.ContainsKey(key)) map[key] = next++;
                        enc[i] = map[key];
                    }
                    return (true, map, enc);
                }
                return (false, null, vals);
            }
            else
            {
                var map = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
                int next = 0;
                var enc = new double[raw.Length];
                for (int i = 0; i < raw.Length; i++)
                {
                    string key = raw[i];
                    if (!map.ContainsKey(key)) map[key] = next++;
                    enc[i] = map[key];
                }
                return (true, map, enc);
            }
        }

        // GA & Evaluation
        private class GAResult
        {
            public double Score;
            public int[] SelectedFeatureIdxs; // indices relative to X columns
            public long TimeMs;
        }

        private GAResult RunGA(int D, int pop, int gens, int kfolds, int knnK, double lambda, int[] globalFeatIdxs)
        {
            // individuals: bool[D]
            var population = new List<bool[]>();
            for (int i = 0; i < pop; i++) population.Add(RandomMask(D));

            bool[] bestMask = null;
            double bestFit = double.NegativeInfinity;
            var sw = Stopwatch.StartNew();

            for (int g = 0; g < gens; g++)
            {
                List<ScoredMask> scored = population.Select(m =>
                {
                    int[] subset = MaskToIdxs(m, globalFeatIdxs);
                    if (subset.Length == 0) subset = new[] { globalFeatIdxs[rnd.Next(globalFeatIdxs.Length)] };

                    double perf = EvaluateSubset(subset, kfolds, knnK);
                    double penalty = lambda * (subset.Length / (double)globalFeatIdxs.Length);
                    double fitness = perf - penalty;

                    return new ScoredMask { m = m, fitness = fitness };
                })
                .OrderByDescending(x => x.fitness)
                .ToList();

                if (scored[0].fitness > bestFit)
                {
                    bestFit = scored[0].fitness;
                    bestMask = (bool[])scored[0].m.Clone();
                }

                var newPop = new List<bool[]>();
                newPop.Add((bool[])scored[0].m.Clone());
                if (pop > 1) newPop.Add((bool[])scored[1].m.Clone());

                while (newPop.Count < pop)
                {
                    var p1 = Tournament(scored);
                    var p2 = Tournament(scored);
                    var (c1, c2) = Crossover(p1, p2);
                    newPop.Add(Mutate(c1, 0.015));
                    if (newPop.Count < pop) newPop.Add(Mutate(c2, 0.015));
                }
                population = newPop;
            }

            sw.Stop();

            var bestIdxs = MaskToIdxs(bestMask, globalFeatIdxs);
            if (bestIdxs.Length == 0) bestIdxs = new[] { globalFeatIdxs[rnd.Next(globalFeatIdxs.Length)] };
            double bestScore = EvaluateSubset(bestIdxs, kfolds, knnK);

            return new GAResult
            {
                Score = bestScore,
                SelectedFeatureIdxs = bestIdxs,
                TimeMs = sw.ElapsedMilliseconds
            };
        }

        private bool[] RandomMask(int D)
        {
            var m = new bool[D];
            for (int i = 0; i < D; i++) m[i] = rnd.NextDouble() < 0.5;
            if (!m.Any(b => b)) m[rnd.Next(D)] = true;
            return m;
        }
        private bool[] Tournament(List<ScoredMask> scored)
        {
            int i1 = rnd.Next(scored.Count), i2 = rnd.Next(scored.Count);
            var winner = (scored[i1].fitness > scored[i2].fitness) ? scored[i1].m : scored[i2].m;
            return (bool[])winner.Clone();
        }

        private (bool[], bool[]) Crossover(bool[] a, bool[] b)
        {
            int D = a.Length;
            int p = rnd.Next(1, D);
            var c1 = new bool[D];
            var c2 = new bool[D];
            for (int i = 0; i < D; i++)
            {
                if (i < p) { c1[i] = a[i]; c2[i] = b[i]; }
                else { c1[i] = b[i]; c2[i] = a[i]; }
            }
            return (c1, c2);
        }
        private bool[] Mutate(bool[] m, double rate)
        {
            var r = (bool[])m.Clone();
            for (int i = 0; i < r.Length; i++)
                if (rnd.NextDouble() < rate) r[i] = !r[i];
            if (!r.Any(x => x)) r[rnd.Next(r.Length)] = true;
            return r;
        }
        private static int[] MaskToIdxs(bool[] mask, int[] globalIdx)
        {
            var list = new List<int>();
            for (int i = 0; i < mask.Length; i++) if (mask[i]) list.Add(globalIdx[i]);
            return list.ToArray();
        }

        //  Evaluation with kNN + KFold 
        private double EvaluateSubset(int[] featIdxs, int kfolds, int knnK)
        {
            // اختر الأعمدة من X
            var Xsub = Project(X, featIdxs);
            // KFold
            int n = Xsub.Length;
            var idx = Enumerable.Range(0, n).OrderBy(_ => rnd.Next()).ToArray();
            int foldSize = Math.Max(1, n / kfolds);
            var scores = new List<double>();
            for (int f = 0; f < kfolds; f++)
            {
                int start = f * foldSize;
                int end = (f == kfolds - 1) ? n : Math.Min(n, start + foldSize);
                var testIdx = idx.Skip(start).Take(end - start).ToArray();
                var trainIdx = idx.Except(testIdx).ToArray();

                var Xtr = trainIdx.Select(i => Xsub[i]).ToArray();
                var ytr = trainIdx.Select(i => y[i]).ToArray();
                var Xte = testIdx.Select(i => Xsub[i]).ToArray();
                var yte = testIdx.Select(i => y[i]).ToArray();

                if (IsClassification)
                {
                    int correct = 0;
                    for (int i = 0; i < Xte.Length; i++)
                    {
                        double pred = KnnPredictCls(Xtr, ytr, Xte[i], knnK);
                        if (Math.Abs(pred - yte[i]) < 1e-9) correct++;
                    }
                    scores.Add(correct / (double)Xte.Length);
                }
                else
                {
                    double sse = 0;
                    for (int i = 0; i < Xte.Length; i++)
                    {
                        double pred = KnnPredictReg(Xtr, ytr, Xte[i], knnK);
                        double err = pred - yte[i];
                        sse += err * err;
                    }
                    double rmse = Math.Sqrt(sse / Xte.Length);
                    scores.Add(1.0 / (1.0 + rmse)); // maximize
                }
            }
            return scores.Average();
        }

        private static double KnnPredictCls(double[][] Xtr, double[] ytr, double[] x, int k)
        {
            var d = new List<(double dist, int idx)>();
            for (int i = 0; i < Xtr.Length; i++)
                d.Add((Dist2(Xtr[i], x), i));
            var top = d.OrderBy(t => t.dist).Take(k).ToArray();
            // تصويت الأكثرية
            return top.GroupBy(t => ytr[t.idx])
                      .OrderByDescending(g => g.Count())
                      .First().Key;
        }
        private static double KnnPredictReg(double[][] Xtr, double[] ytr, double[] x, int k)
        {
            var d = new List<(double dist, int idx)>();
            for (int i = 0; i < Xtr.Length; i++)
                d.Add((Dist2(Xtr[i], x), i));
            var top = d.OrderBy(t => t.dist).Take(k).ToArray();
            return top.Select(t => ytr[t.idx]).Average();
        }
        private static double Dist2(double[] a, double[] b)
        {
            double s = 0;
            for (int i = 0; i < a.Length; i++)
            {
                double dx = a[i] - b[i];
                s += dx * dx;
            }
            return s;
        }

        private static double[][] Project(double[][] X, int[] cols)
        {
            int n = X.Length, d = cols.Length;
            var Y = new double[n][];
            for (int i = 0; i < n; i++)
            {
                Y[i] = new double[d];
                for (int j = 0; j < d; j++) Y[i][j] = X[i][cols[j]];
            }
            return Y;
        }

        //  Univariate Top-K
        private int[] UnivariateTopK(int[] featIdxs, int K)
        {
            // تصنيف: point-biserial إذا ثنائي، وإلا Pearson مع ترميز الأهداف
            bool binary = IsClassification && y.Distinct().Count() == 2;

            var scores = new List<(int idx, double sc)>();
            foreach (var fi in featIdxs)
            {
                var col = X.Select(r => r[fi]).ToArray();
                double sc;
                if (!IsClassification)
                {
                    sc = Math.Abs(Pearson(col, y));
                }
                else if (binary)
                {
                    sc = Math.Abs(PointBiserial(col, y)); // y=0/1 مثالي، وإن كانت غير 0/1 فالمقياس ما يزال يعمل
                }
                else
                {
                    sc = Math.Abs(Pearson(col, y)); // fallback
                }
                if (double.IsNaN(sc)) sc = 0;
                scores.Add((fi, sc));
            }

            return scores.OrderByDescending(s => s.sc).Take(Math.Min(K, scores.Count)).Select(s => s.idx).ToArray();
        }

        // Pearson correlation
        private static double Pearson(double[] a, double[] b)
        {
            double ma = a.Average(), mb = b.Average();
            double num = 0, da = 0, db = 0;
            for (int i = 0; i < a.Length; i++)
            {
                double xa = a[i] - ma, xb = b[i] - mb;
                num += xa * xb;
                da += xa * xa;
                db += xb * xb;
            }
            double den = Math.Sqrt(da * db);
            if (den <= 1e-12) return 0;
            return num / den;
        }
        // Point-biserial (y ثنائي)
        private static double PointBiserial(double[] x, double[] y)
        {
            // y يفترض قيمتين، قد تكون 0/1 أو قيم أخرى — نقسم حسب أول قيمة
            var c0 = y[0];
            var g1 = x.Where((_, i) => Math.Abs(y[i] - c0) < 1e-9).ToArray();
            var g2 = x.Where((_, i) => Math.Abs(y[i] - c0) >= 1e-9).ToArray();
            if (g1.Length == 0 || g2.Length == 0) return 0;
            double m1 = g1.Average(), m2 = g2.Average();
            double s = Std(x);
            if (s <= 1e-12) return 0;
            double p = g1.Length / (double)x.Length, q = 1 - p;
            return ((m1 - m2) / s) * Math.Sqrt(p * q);
        }
        private static double Std(double[] a)
        {
            double m = a.Average(), s = 0;
            for (int i = 0; i < a.Length; i++) { double d = a[i] - m; s += d * d; }
            return Math.Sqrt(s / Math.Max(1, a.Length - 1));
        }


    }
}

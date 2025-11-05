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




    }
}

<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="Default.aspx.cs" Inherits="WebApp.Default" %>

  <!DOCTYPE html>
  <html xmlns="http://www.w3.org/1999/xhtml" lang="ar" dir="rtl">

  <head runat="server">
    <meta charset="utf-8" />
    <title>اختيار الميزات بالخوارزمية الجينية | Feature Selection (GA)</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <!-- Bootstrap -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" />
    <!-- Font Awesome -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet" />
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>

    <style>
      body {
        background: #f6f8fb
      }

      .card {
        border: none;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, .06)
      }

      .kpi .value {
        font-weight: 800;
        font-size: 1.25rem
      }

      .chart-wrap {
        background: #fff;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, .06)
      }

      .table thead th {
        white-space: nowrap
      }

      code,
      .ltr {
        direction: ltr
      }

      .fa-ul {
        margin-right: 0
      }

      .fa-li {
        right: auto;
        left: unset
      }
    </style>
  </head>

  <body class="bg-light">

    <form id="form1" runat="server" class="container py-4">

      <!-- Header -->
      <div class="card mb-4 p-3 bg-primary text-white">
        <div class="d-flex align-items-center gap-3 flex-wrap">
          <i class="fa-solid fa-dna fa-2x"></i>
          <div>
            <h3 class="m-0 fw-bold">اختيار الميزات باستخدام الخوارزمية الجينية (GA)</h3>
            <small class="opacity-75">ابحث عن المجموعة المثلى من ميزات <code class="ltr">data.csv</code> التي تعظّم أداء
              النموذج وتقلّل عدد الميزات.</small>
          </div>
        </div>
      </div>

  <!-- Upload & Config -->
  <div class="card mb-4">
    <div class="card-header bg-info text-white">
     رفع البيانات والإعداد
    </div>
    <div class="card-body">
      <div class="row g-3 align-items-center">
        <div class="col-12 col-md-6">
          <label class="form-label"><i class="fa-solid fa-file-csv"></i> ملف CSV</label>
          <asp:FileUpload ID="fuCsv" runat="server" CssClass="form-control form-control-sm" />
          <div class="form-text">السطر الأول يُفضّل أن يكون عناوين أعمدة. الفاصل: <code class="ltr">,</code> أو <code class="ltr">;</code> أو <code class="ltr">\t</code></div>
        </div>
        <div class="col-12 col-md-6 d-flex gap-2">
          <asp:Button ID="btnLoadCsv" runat="server" CssClass="btn btn-secondary"
            Text="تحميل ومعاينة" OnClick="btnLoadCsv_Click" />
          <asp:Button ID="btnUseSample" runat="server" CssClass="btn btn-outline-secondary"
            Text="استخدام بيانات عينة" OnClick="btnUseSample_Click" />
        </div>
      </div>

      <hr />

      <div class="row g-3">
        <div class="col-md-4">
          <label class="form-label"><i class="fa-solid fa-bullseye"></i> عمود الهدف (Target)</label>
          <asp:DropDownList ID="ddlTarget" runat="server" CssClass="form-select form-select-sm"></asp:DropDownList>
        </div>
        <div class="col-md-4">
          <label class="form-label"><i class="fa-solid fa-tags"></i> نوع المشكلة</label>
          <asp:DropDownList ID="ddlProblemType" runat="server" CssClass="form-select form-select-sm">
            <asp:ListItem Text="تصنيف" Value="cls" />
            <asp:ListItem Text="انحدار" Value="reg" />
          </asp:DropDownList>
          <div class="form-text">يمكنك تركها كما هي وسيتم الاستدلال تلقائياً بعد التحميل.</div>
        </div>
        <div class="col-md-4">
          <label class="form-label"><i class="fa-solid fa-filter-circle-dollar"></i> عقوبة عدد الميزات λ</label>
          <asp:TextBox ID="txtPenalty" runat="server" CssClass="form-control form-control-sm ltr" TextMode="Number" Text="0.02" />
          <div class="form-text">قيمة صغيرة (0.01–0.05). كلما زادت زاد تقليل الميزات.</div>
        </div>
      </div>

      <div class="row g-3 mt-1">
        <div class="col-md-4">
          <label class="form-label"><i class="fa-solid fa-people-group"></i> حجم السكان (GA)</label>
          <asp:TextBox ID="txtPop" runat="server" CssClass="form-control form-control-sm ltr" TextMode="Number" Text="30" />
        </div>
        <div class="col-md-4">
          <label class="form-label"><i class="fa-solid fa-infinity"></i> عدد الأجيال (GA)</label>
          <asp:TextBox ID="txtGen" runat="server" CssClass="form-control form-control-sm ltr" TextMode="Number" Text="25" />
        </div>
        <div class="col-md-4">
          <label class="form-label"><i class="fa-solid fa-grip-lines"></i> عدد الطيات K-Fold</label>
          <asp:TextBox ID="txtKFolds" runat="server" CssClass="form-control form-control-sm ltr" TextMode="Number" Text="3" />
        </div>
      </div>

      <div class="row g-3 mt-1">
        <div class="col-md-4">
          <label class="form-label"><i class="fa-solid fa-list-ol"></i> Top-K (الطريقة التقليدية)</label>
          <asp:TextBox ID="txtTopK" runat="server" CssClass="form-control form-control-sm ltr" TextMode="Number" Text="30" />
        </div>
        <div class="col-md-4">
          <label class="form-label"><i class="fa-solid fa-user-ninja"></i> k-NN (k)</label>
          <asp:TextBox ID="txtKnnK" runat="server" CssClass="form-control form-control-sm ltr" TextMode="Number" Text="5" />
        </div>
        <div class="col-md-4 d-flex align-items-end">
          <asp:Button ID="btnRun" runat="server" CssClass="btn btn-primary w-100"
            Text="تشغيل وتحليل" OnClick="btnRun_Click" />
        </div>
      </div>

      <div class="mt-3">
        <asp:Literal ID="litLoadInfo" runat="server" />
      </div>
    </div>
  </div>

  <!-- KPIs -->
  <div class="row g-3 mb-4">
    <div class="col-6 col-md-3">
      <div class="card kpi p-3">
        <div class="text-muted">عدد العينات</div>
        <div class="value" id="kpiSamples">—</div>
      </div>
    </div>
    <div class="col-6 col-md-3">
      <div class="card kpi p-3">
        <div class="text-muted">عدد الميزات</div>
        <div class="value" id="kpiFeatures">—</div>
      </div>
    </div>
    <div class="col-6 col-md-3">
      <div class="card kpi p-3">
        <div class="text-muted">أفضل أداء (GA)</div>
        <div class="value" id="kpiBestScore">—</div>
      </div>
    </div>
    <div class="col-6 col-md-3">
      <div class="card kpi p-3">
        <div class="text-muted">عدد ميزات GA</div>
        <div class="value" id="kpiGaCount">—</div>
      </div>
    </div>
  </div>

  <!-- Results & Charts -->
  <div class="row g-3">
    <div class="col-lg-6">
      <div class="chart-wrap">
        <h6 class="mb-2"><i class="fa-solid fa-chart-column me-1"></i> مقارنة الأداء</h6>
        <canvas id="cmpChart" height="240"></canvas>
        <small class="text-muted d-block mt-1">Baseline-All vs. Univariate Top-K vs. GA</small>
      </div>
    </div>
    <div class="col-lg-6">
      <div class="chart-wrap">
        <h6 class="mb-2"><i class="fa-solid fa-chart-simple me-1"></i> أعلى الميزات (GA)</h6>
        <canvas id="featChart" height="240"></canvas>
        <small class="text-muted d-block mt-1">أكثر 20 ميزة ظهرت في مجموعة GA.</small>
      </div>
    </div>

    <div class="col-12">
      <div class="card p-3">
        <h6 class="mb-2"><i class="fa-solid fa-list-check me-1"></i> تفاصيل النتائج</h6>
        <asp:Literal ID="litResult" runat="server" />
        <!-- يتم حقن سكربتات الرسوم من الخلفية في هذا اللترال -->
        <asp:Literal ID="litCharts" runat="server" />
      </div>
    </div>

    <div class="col-12">
      <div class="card p-3">
        <h6 class="mb-2"><i class="fa-regular fa-clock me-1"></i> السجل</h6>
        <asp:Literal ID="litHistory" runat="server" />
      </div>
    </div>
  </div>

</form>

      <!-- Bootstrap JS -->
      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
  </body>

  </html>
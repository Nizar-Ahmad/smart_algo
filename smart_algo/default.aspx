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
    body{background:#f6f8fb}
    .card{border:none;border-radius:14px;box-shadow:0 6px 18px rgba(0,0,0,.06)}
    .kpi .value{font-weight:800;font-size:1.25rem}
    .chart-wrap{background:#fff;border-radius:14px;padding:16px;box-shadow:0 6px 18px rgba(0,0,0,.06)}
    .table thead th{white-space:nowrap}
    code, .ltr {direction:ltr}
    .fa-ul{margin-right:0}
    .fa-li{right:auto;left:unset}
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
        <small class="opacity-75">ابحث عن المجموعة المثلى من ميزات <code class="ltr">data.csv</code> التي تعظّم أداء النموذج وتقلّل عدد الميزات.</small>
      </div>
    </div>
  </div>


<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>

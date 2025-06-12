S = 0.8;
N = 0.6;
lambda1 = 1.0;
lambda2 = 1.0;
Rcognitive = 0.3;
Refficiency = 0.2;
Pbiased = 0.75;
regularization = Exp[-(lambda1*Rcognitive + lambda2*Refficiency)];
Psi[alpha_] := (alpha*S + (1 - alpha)*N)*regularization*Pbiased;
points = Table[{a, Psi[a]}, {a, 0, 1, 0.25}];
Plot[
  Psi[a],
  {a, 0, 1},
  PlotRange -> All,
  AxesLabel -> {"\[Alpha]", "\[Psi](x)"},
  Epilog -> {
    Red, PointSize[Medium],
    Point[points],
    Map[
      ({x, y} = #; Text[NumberForm[y, {3, 3}], {x + 0.02, y + 0.02}]) &,
      points
    ]
  },
  GridLines -> Automatic,
  PlotLabel -> "\[Psi](x) vs \[Alpha]",
  ImageSize -> Large
]
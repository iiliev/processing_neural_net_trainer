void runAllTests(){
  println(String.format(               "linearForward error: %.4f",                 linearForward())); 
  println(String.format(     "linearActivationForward error: %.4f",       linearActivationForward())); 
  println(String.format(                         "laf error: %.4f",                           laf())); 
  println(String.format(                 "computeCost error: %.4f",                   computeCost())); 
  println(String.format(           "testLModelForward error: %.4f",             testLModelForward()));
  println(String.format(          "testLinearBackward error: %.4f",            testLinearBackward()));
  println(String.format("testLinearActivationBackward error: %.4f",  testLinearActivationBackward()));
  println(String.format(          "testLModelBackward error: %.4f",            testLModelBackward()));
  println(String.format(        "testUpdateParameters error: %.4f",          testUpdateParameters()));  
  println(String.format(       "testLModelForwardTanh error: %.4f",         testLModelForwardTanh()));  
  println(String.format(      "testLModelBackwardTanh error: %.4f",        testLModelBackwardTanh()));  
}

float linearForward(){
  float[][] a = new float[][]{{ 1.62434536, -0.61175641},
                              {-0.52817175, -1.07296862},
                              { 0.86540763, -2.3015387 }}; 
  float[][] w = new float[][]{{ 1.74481176, -0.7612069 ,  0.3190391 }};
  float[][] b = new float[][]{{ -0.24937038 }};
  Matrix A = new Matrix(a);
  Matrix W = new Matrix(w);
  Matrix B = new Matrix(b);
  
  Matrix Z = multiply( W, A ).broadcastAndAdd(B);

  Matrix Zcorrect = new Matrix( new float[][]{{ 3.26295337, -1.23429987 }} );
  return RMSE( Z, Zcorrect );
}

float linearActivationForward(){
  float[][] a = new float[][]{{-0.41675785, -0.05626683},
                              {-2.1361961 ,  1.64027081},
                              {-1.79343559, -0.84174737}};
  float[][] w = new float[][]{{ 0.50288142, -1.24528809, -1.05795222 }};
  float[][] b = new float[][]{{ -0.90900761 }};
  Matrix A = new Matrix(a);
  Matrix W = new Matrix(w);
  Matrix B = new Matrix(b);
  
  float error = 0;
  Matrix Z = multiply( W, A ).broadcastAndAdd(B);
  Z.activateSigm();
  Matrix Zcorrect = new Matrix( new float[][]{{0.96890023, 0.11013289}} );
  error += RMSE( Z, Zcorrect );
  
  Z = multiply( W, A ).broadcastAndAdd(B);
  Z.activateRelu();
  Zcorrect = new Matrix( new float[][]{{3.43896131, 0.}});
  error += RMSE( Z, Zcorrect );
  
  return error;
}

float laf(){ //same as linearActivationForward above
  float[][] a = new float[][]{{-0.41675785, -0.05626683},
                              {-2.1361961 ,  1.64027081},
                              {-1.79343559, -0.84174737}};
  float[][] w = new float[][]{{ 0.50288142, -1.24528809, -1.05795222 }};
  float[][] b = new float[][]{{ -0.90900761 }};
  Matrix A = new Matrix(a);
  Matrix W = new Matrix(w);
  Matrix B = new Matrix(b);
  
  
  Map<String,Matrix> m1 = linearActivationForward( A, W, B, "sigmoid" );
  float error = 0;
  Matrix Acorrect = new Matrix( new float[][]{{0.96890023, 0.11013289}} );
  error += RMSE( m1.get("A"), Acorrect );
  
  m1 = linearActivationForward( A, W, B, "relu" );
  Acorrect = new Matrix( new float[][]{{3.43896131, 0.}});
  error += RMSE( m1.get("A"), Acorrect );
  
  return error;
}

float computeCost(){
  float [] al = new float[]{0.8, 0.9, 0.4};  
  float [] y  = new float[]{1,1,0};  
  return cost(al,y) - 0.2797765635793423;
}

Map<String,Matrix> initParamsForward(){
  Map<String,Matrix> params = new HashMap<String,Matrix>();

 float [][]w1 = new float[][]{ { 0.35480861,  1.81259031, -1.3564758 , -0.46363197,  0.82465384}, //5 dimensional input
                               {-1.17643148,  1.56448966,  0.71270509, -0.1810066 ,  0.53419953}, //4 dimensional first layer
                               {-0.58661296, -1.48185327,  0.85724762,  0.94309899,  0.11444143},
                               {-0.02195668, -2.12714455, -0.83440747, -0.46550831,  0.23371059}};
  float [][]b1 = new float[][]{{ 1.38503523 },
                               {-0.51962709 },  //4 dimensional first layer
                               {-0.78015214 },
                               { 0.95560959 }};
  float [][]w2 = new float[][]{{-0.12673638, -1.36861282,  1.21848065, -0.85750144}, //4 dimensional first layer
                               {-0.56147088, -1.0335199 ,  0.35877096,  1.07368134}, //3 dimensional second layer
                               {-0.37550472,  0.39636757, -0.47144628,  2.33660781}};
  float [][]b2 = new float[][]{{ 1.50278553 },
                               {-0.59545972 },    //3 dimensional second layer
                               { 0.52834106 }};
  float [][]w3 = new float[][]{{ 0.9398248 ,  0.42628539, -0.75815703}};             //3 dimensional second layer, 1 dim output
  float [][]b3 = new float[][]{{-0.16236698 }};  
  params.put("W1", new Matrix(w1));
  params.put("B1", new Matrix(b1));
  params.put("W2", new Matrix(w2));
  params.put("B2", new Matrix(b2));
  params.put("W3", new Matrix(w3));
  params.put("B3", new Matrix(b3));
  return params;
}

float testLModelForward(){
  float [][]x = new float[][]{{-0.31178367,  0.72900392,  0.21782079, -0.8990918 }, //5 dimensional input
                              {-2.48678065,  0.91325152,  1.12706373, -1.51409323}, //4 training examples
                              { 1.63929108, -0.4298936 ,  2.63128056,  0.60182225},
                              {-0.33588161,  1.23773784,  0.11112817,  0.12915125},
                              { 0.07612761, -0.15512816,  0.63422534,  0.810655  }};
  Matrix X = new Matrix(x);
  Map<String,Matrix>[] caches = lModelForward( X, initParamsForward(), new String[]{"relu","relu","sigmoid"});

  Matrix Acorrect = new Matrix( new float[][]{{0.03921668, 0.70498921, 0.19734387, 0.04728177}} );
  float error = RMSE( caches[caches.length-1].get("A"), Acorrect );
  return error;
}

float testLinearBackward(){
  float[][] dz = new float[][]{ { 1.62434536, -0.61175641, -0.52817175, -1.07296862},
                                { 0.86540763, -2.3015387 ,  1.74481176, -0.7612069 },
                                { 0.3190391 , -0.24937038,  1.46210794, -2.06014071}};
   float[][] a_prev = new float[][]{ {-0.3224172 , -0.38405435,  1.13376944, -1.09989127},
                                     {-0.17242821, -0.87785842,  0.04221375,  0.58281521},
                                     {-1.10061918,  1.14472371,  0.90159072,  0.50249434},
                                     { 0.90085595, -0.68372786, -0.12289023, -0.93576943},
                                     {-0.26788808,  0.53035547, -0.69166075, -0.39675353}};
  float[][] w = new float[][]{ {-0.6871727 , -0.84520564, -0.67124613, -0.0126646 , -1.11731035},
                               { 0.2344157 ,  1.65980218,  0.74204416, -0.19183555, -0.88762896},
                               {-0.74715829,  1.6924546 ,  0.05080775, -0.63699565,  0.19091548}};
  float[][] b = new float[][]{ {2.10025514},
                               {0.12015895},
                               {0.61720311}};
  Matrix dZ     = new Matrix( dz );
  Matrix A_prev = new Matrix( a_prev );
  Matrix W      = new Matrix(w);
  Matrix B      = new Matrix(b); //not used, the Python code used to assert dimensions with it
  
  Map<String,Matrix> result = linearBackward( dZ,  A_prev,  W );
  //printMap(result);
  
  Matrix dA_prev = new Matrix( new float[][]{{-1.15171336,  0.06718465, -0.3204696 ,  2.09812712},
                                             { 0.60345879, -3.72508701,  5.81700741, -3.84326836},
                                             {-0.4319552 , -1.30987417,  1.72354705,  0.05070578},
                                             {-0.38981415,  0.60811244, -1.25938424,  1.47191593},
                                             {-2.52214926,  2.67882552, -0.67947465,  1.48119548}} );
  float error = RMSE( result.get("dA_prev"), dA_prev );
  
  Matrix dW = new Matrix( new float[][]{{0.07313866, -0.0976715 , -0.87585828,  0.73763362,  0.00785716},
                                             {0.85508818,  0.37530413, -0.59912655,  0.71278189, -0.58931808},
                                             {0.97913304, -0.24376494, -0.08839671,  0.55151192, -0.10290907}});  
  error += RMSE( result.get("dW"), dW );                                
  
  Matrix dB = new Matrix( new float[][]{{  -0.14713786},{-0.11313155},{-0.13209101}});
  error += RMSE( result.get("dB"), dB );                                

  return error;  
}

float testLinearActivationBackward(){
  float[][]     da = new float[][]{{-0.41675785, -0.05626683}};
  float[][] a_prev = new float[][]{ {-2.1361961 ,  1.64027081},
                                    {-1.79343559, -0.84174737},
                                    { 0.50288142, -1.24528809}};
  float[][]      w = new float[][]{{-1.05795222, -0.90900761,  0.55145404}};
  float[][]      z = new float[][]{{ 0.04153939, -1.11792545}};  
  Matrix dA     = new Matrix( da );
  Matrix A_prev = new Matrix( a_prev );
  Matrix W      = new Matrix( w );
  Matrix Z      = new Matrix( z );

  float error = 0;
  Map<String,Matrix> result = linearActivationBackward( dA, Z, A_prev, W, "sigmoid");
  Matrix dA_prev = new Matrix( new float[][]{{ 0.11017994, 0.01105339},{0.09466817, 0.00949723},{-0.05743092, -0.00576154}});
  error += RMSE( result.get("dA_prev"), dA_prev );
  Matrix dW = new Matrix( new float[][]{{0.10266786, 0.09778551, -0.01968084}});
  error += RMSE( result.get("dW"), dW );
  Matrix dB = new Matrix( new float[][]{{-0.05729622}});
  error += RMSE( result.get("dB"), dB );
  
  result = linearActivationBackward( dA, Z, A_prev, W, "relu");
  dA_prev = new Matrix( new float[][]{{ 0.44090989, 0.},{0.37883606, 0.},{-0.2298228, 0.}});
  error += RMSE( result.get("dA_prev"), dA_prev );
  dW = new Matrix( new float[][]{{0.44513824, 0.37371418, -0.10478989}});
  error += RMSE( result.get("dW"), dW );
  dB = new Matrix( new float[][]{{-0.20837892}});
  error += RMSE( result.get("dB"), dB );
  return error;
}

float testLModelBackward(){
  float[][] al = new float[][]{{1.78862847, 0.43650985}};  
  float[][]  y = new float[][]{{1         , 0         }};  
  Matrix AL = new Matrix(al);
  Matrix  Y = new Matrix( y);
  
  Map<String,Matrix>[] caches = (Map<String,Matrix>[]) new Map[2];

  Map<String,Matrix> entry = new HashMap<String,Matrix>(); 
  
  float[][] a_prev = new float[][]{{ 0.09649747, -1.8634927 },
                                   {-0.2773882 , -0.35475898},
                                   {-0.08274148, -0.62700068},
                                   {-0.04381817, -0.47721803}};
           
  float[][] w = new float[][]{ {-1.31386475,  0.88462238,  0.88131804,  1.70957306},
                               { 0.05003364, -0.40467741, -0.54535995, -1.54647732},
                               { 0.98236743, -1.10106763, -1.18504653, -0.2056499 }};
  float[][] z = new float[][]{{-0.7129932 ,  0.62524497},
                              {-0.16051336, -0.76883635},
                              {-0.23003072,  0.74505627}};
  entry.put("Z"     , new Matrix(     z));
  entry.put("A_prev", new Matrix(a_prev));
  entry.put("W"     , new Matrix(     w));
  caches[0] = entry;      

  entry = new HashMap<String,Matrix>();
  a_prev = new float[][]{{ 1.97611078, -1.24412333},
                         {-0.62641691, -0.80376609},
                         {-2.41908317, -0.92379202}};
  w = new float[][]{{-1.02387576,  1.12397796, -0.13191423}};
  z = new float[][]{{ 0.64667545, -0.35627076}};        
  entry.put("Z"     , new Matrix(     z));
  entry.put("A_prev", new Matrix(a_prev));
  entry.put("W"     , new Matrix(     w));
  caches[1] = entry;      
  
  Map<String,Matrix> grads = lModelBackward( AL, Y, caches, new String[]{"relu","sigmoid"} );
  //printMap(grads);
  float error = 0;
  Matrix dA1 = new Matrix( new float[][]{{ 0.12913162, -0.44014127},{-0.14175655, 0.48317296},{0.01663708, -0.05670698}});
  error += RMSE( grads.get("dA1"), dA1 );

  Matrix dW1 = new Matrix( new float[][]{ {0.41010002, 0.07807203, 0.13798444, 0.10502167},
                                          {0.,         0.,         0.,         0.        },
                                          {0.05283652, 0.01005865, 0.01777766, 0.0135308}});
  error += RMSE( grads.get("dW1"), dW1 );

  Matrix dB1 = new Matrix( new float[][]{{-0.22007063},{0.},{-0.02835349}});
  error += RMSE( grads.get("dB1"), dB1 );
  return error;
}

float testUpdateParameters(){
  Map<String,Matrix> params = new HashMap<String,Matrix>();
  float[][] w1 = new float[][]{{-0.41675785, -0.05626683, -2.1361961 ,  1.64027081},
                               {-1.79343559, -0.84174737,  0.50288142, -1.24528809},
                               {-1.05795222, -0.90900761,  0.55145404,  2.29220801}};
  float[][] b1 = new float[][]{{0.04153939},
                               {-1.11792545},
                               { 0.53905832}};
  float[][] w2 = new float[][]{{-0.5961597 , -0.0191305 ,  1.17500122}};
  float[][] b2 = new float[][]{{-0.74787095}};  
  params.put("W1", new Matrix( w1 ));
  params.put("B1", new Matrix( b1 ));
  params.put("W2", new Matrix( w2 ));
  params.put("B2", new Matrix( b2 ));
  
  Map<String,Matrix> grads = new HashMap<String,Matrix>();
  float[][] dw1 = new float[][]{ { 1.78862847,  0.43650985,  0.09649747, -1.8634927 },
                                 {-0.2773882 , -0.35475898, -0.08274148, -0.62700068},
                                 {-0.04381817, -0.47721803, -1.31386475,  0.88462238}};
  float[][] db1 = new float[][]{ {0.88131804},
                                 {1.70957306},
                                 {0.05003364}};
  float[][] dw2 = new float[][]{{-0.40467741, -0.54535995, -1.54647732}};
  float[][] db2 = new float[][]{{0.98236743}};
  grads.put("dW1", new Matrix( dw1 ));
  grads.put("dB1", new Matrix( db1 ));
  grads.put("dW2", new Matrix( dw2 ));
  grads.put("dB2", new Matrix( db2 ));
  
  updateParameters( params, grads, .1 );

  float error = 0;
  Matrix W1 = new Matrix( new float[][]{{-0.59562069, -0.09991781, -2.14584584,  1.82662008},
                                        {-1.76569676, -0.80627147,  0.51115557, -1.18258802},
                                        {-1.0535704,  -0.86128581,  0.68284052,  2.20374577}});
  error += RMSE( params.get("W1"), W1 );

  Matrix B1 = new Matrix( new float[][]{{-0.04659241},{-1.28888275},{0.53405496}});
  error += RMSE( params.get("B1"), B1 );

  Matrix W2 = new Matrix( new float[][]{ {-0.55569196, 0.0354055, 1.32964895}});
  error += RMSE( params.get("W2"), W2 );

  Matrix B2 = new Matrix( new float[][]{{-0.84610769}});
  error += RMSE( params.get("B2"), B2 );
  
  return error;
}

float testLModelForwardTanh(){
  float [][]x = new float[][]{{1.62434536, -0.61175641, -0.52817175},
                              {-1.07296862,  0.86540763, -2.3015387}};
  float [][]w1 = new float[][]{{-0.00416758, -0.00056267},
                               {-0.02136196,  0.01640271},
                               {-0.01793436, -0.00841747},
                               { 0.00502881, -0.01245288}};                              
  float [][]w2 = new float[][]{{-0.01057952, -0.00909008,  0.00551454,  0.02292208}};
  float [][]b1 = new float[][]{{ 1.74481176},{-0.7612069},{ 0.3190391},{-0.24937038}};
  float [][]b2 = new float[][]{{ -1.3}};   
  Matrix X = new Matrix(x);
  Map<String,Matrix> params = new HashMap<String,Matrix>();
  params.put("W1", new Matrix(w1));  
  params.put("W2", new Matrix(w2));  
  params.put("B1", new Matrix(b1));  
  params.put("B2", new Matrix(b2));  
  
  Map<String,Matrix>[] caches = lModelForward( X, params, new String[]{"tanh", "sigmoid"});
  Matrix A = caches[caches.length - 1].get("A");
  Matrix Acorrect = new Matrix( new float[][]{{0.21292656, 0.21274673, 0.21295976}});
  return RMSE( Acorrect, A );
}        

float testLModelBackwardTanh(){
  float[][] al = new float[][]{{0.5002307, 0.49985831, 0.50023963}};  
  float[][]  y = new float[][]{{1, 0, 1}};  
  Matrix AL = new Matrix(al);
  Matrix  Y = new Matrix( y);
    
  Map<String,Matrix>[] caches = (Map<String,Matrix>[]) new Map[2];

  Map<String,Matrix> entry = new HashMap<String,Matrix>();   
  float[][] a_prev = new float[][]{ {-0.00616578,  0.0020626 ,  0.00349619},
                                    {-0.05225116,  0.02725659, -0.02646251},
                                    {-0.02009721,  0.0036869 ,  0.02883756},
                                    { 0.02152675, -0.01385234,  0.02599885}};
           
  float[][] w = new float[][]{ {-0.01057952, -0.00909008,  0.00551454,  0.02292208}};
  
  float[][] z = new float[][]{{0.00092281, -0.00056678,  0.00095853}};
  entry.put("Z"     , new Matrix(     z));
  entry.put("A_prev", new Matrix(a_prev));
  entry.put("W"     , new Matrix(     w));
  caches[1] = entry;

  Matrix dZ = new Matrix( AL ).subtract( Y ); //just a demonstration that the complex calculation in linearActivationBackward
  dZ.printM();                                //is the same as AL - Y, why is the one in linearActivationBackward needed?
  Matrix dAL = elementWiseDivide(add( Y , -1 ).multiply(-1), //this is from lModelBackward, why is it necessary if AL-Y works?
                                 add( AL, -1 ).multiply(-1)).subtract(elementWiseDivide( Y, AL )); // (1-Y)/(1-AL) - Y/AL
  sigmoidBackward( dAL, caches[1].get("Z") ).printM();
 
  entry = new HashMap<String,Matrix>();   
  a_prev = new float[][]{{1.62434536, -0.61175641, -0.52817175},
                         {-1.07296862,  0.86540763, -2.3015387}};
  w = new float[][]{{-0.00416758, -0.00056267},
                    {-0.02136196,  0.01640271},
                    {-0.01793436, -0.00841747},
                    { 0.00502881, -0.01245288}};  
  z = new float[][]{{-0.00616586,  0.0020626 ,  0.0034962 },
                    {-0.05229879,  0.02726335, -0.02646869},
                    {-0.02009991,  0.00368692,  0.02884556},
                    { 0.02153007, -0.01385322,  0.02600471}};
  entry.put("Z"     , new Matrix(     z));
  entry.put("A_prev", new Matrix(a_prev));
  entry.put("W"     , new Matrix(     w));
  caches[0] = entry;
  
  Map<String,Matrix> grads = lModelBackward( AL, Y, caches, new String[]{"tanh", "sigmoid"} );

  float error = 0;
  Matrix dW1 = new Matrix( new float[][]{ { 0.00301023, -0.00747267},
                                          {0.00257968, -0.00641288},
                                          {-0.00156892,  0.003893},
                                          {-0.00652037,  0.01618243}});
  error += RMSE( grads.get("dW1"), dW1 );

  Matrix dW2 = new Matrix( new float[][]{ {0.00078841, 0.01765429, -0.00084166, -0.01022527}});
  error += RMSE( grads.get("dW2"), dW2 );

  Matrix dB1 = new Matrix( new float[][]{{0.00176201},{0.00150995},{-0.00091736},{-0.00381422}});
  error += RMSE( grads.get("dB1"), dB1 );
  
  return error;
}

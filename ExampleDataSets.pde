float scaleVel( float vel ){
  float velLimit  =   10.;
  if( vel > velLimit ) return 1;
  return vel/velLimit;
}
float scaleDst( float dst ){
  float dstLimit  =  600.;
  return dst/dstLimit;
}
float scaleAng( float ang ){
  float angLimit  =   PI;
  return ang/angLimit;
}

ArrayTrainer xorTrainer( NeuralNetwork nn ){
  ArrayList<Matrix> inputs = new ArrayList<Matrix>();  
  inputs.add( new Matrix( new float[]{0,0} ) );
  inputs.add( new Matrix( new float[]{0,1} ) );
  inputs.add( new Matrix( new float[]{1,0} ) );
  inputs.add( new Matrix( new float[]{1,1} ) );
    
  ArrayList<Matrix> targets = new ArrayList<Matrix>();  
  targets.add( new Matrix( new float[]{0} ) );
  targets.add( new Matrix( new float[]{1} ) );
  targets.add( new Matrix( new float[]{1} ) );
  targets.add( new Matrix( new float[]{0} ) );
  
  return new ArrayTrainer( inputs, targets, nn );
}

MatrixTrainer xorTrainerM( NeuralNetwork nn ){
  float[][] i = new float[][]{{ 0, 0, 1, 1 },
                              { 0, 1, 0, 1 }};
  float[][] t = new float[][]{{ 0, 1, 1, 0 }};
  return new MatrixTrainer( new Matrix(i), new Matrix(t), nn );
}

ArrayTrainer carData3AngleTrainer( NeuralNetwork nn ){
  float velLimit  =   10.;
  float dstLimit  =  800.;
  float angLimit  =   PI; //180 degrees
  ArrayList<Matrix> inputs  = new ArrayList<Matrix>();
  ArrayList<Matrix> targets = new ArrayList<Matrix>();
  float[] velocityT = new float[]{ velLimit*.012, velLimit*.037, velLimit*.11, velLimit*.33, velLimit };
  float[] anglesT   = new float[]{ angLimit*.012, angLimit*.037, angLimit*.11, angLimit*.33, angLimit };
  float[] distT     = new float[]{ dstLimit*.012, dstLimit*.037, dstLimit*.11, dstLimit*.33, dstLimit };
  for( int v = 0; v < velocityT.length; v++ )
    for( int a0 = 0; a0 < anglesT.length; a0++ )
      for( int d0 = 0; d0 < distT.length; d0++ )
        for( int a1 = 0; a1 < anglesT.length; a1++ )
          for( int d1 = 0; d1 < distT.length; d1++ )
            for( int a2 = 0; a2 < anglesT.length; a2++ ){
              float[] input = { scaleVel(velocityT[v]), 
                                scaleAng(anglesT[a0]), 
                                scaleDst(distT[d0]), 
                                scaleAng(anglesT[a1]), 
                                scaleDst(distT[d1]), 
                                scaleAng(anglesT[a2]) };
              //println(String.format("%.2f, %.2f, %.2f, %.2f, %.2f, %.2f",input[0],input[1],input[2],input[3],input[4],input[5]));
              float[] output;
              if( velocityT[v] < 1. )
                if( anglesT[a0] <= .10 ) output = new float[]{1,0,0,0};
                else                     output = new float[]{1,0,1,0};
              else
                if( anglesT[a0] <= .10 ) output = new float[]{0,0,0,0};
                else                     output = new float[]{0,0,1,0};
              inputs .add( new Matrix( input));
              targets.add( new Matrix(output));
            }
  return new ArrayTrainer( inputs, targets, nn );
}

ArrayTrainer carData2AngleTrainer( NeuralNetwork nn ){
  float velLimit  =   10.;
  float dstLimit  =  800.;
  float angLimit  =   PI; //180 degrees
  ArrayList<Matrix> inputs  = new ArrayList<Matrix>();
  ArrayList<Matrix> targets = new ArrayList<Matrix>();
  float[] velocityT = new float[]{                velLimit*.037, velLimit*.11, velLimit*.33, velLimit };
  float[] anglesT   = new float[]{  angLimit*.012,  angLimit*.037,  angLimit*.11,  angLimit*.33,  angLimit };
  float[] distT     = new float[]{ dstLimit*.012, dstLimit*.037, dstLimit*.11, dstLimit*.33, dstLimit };
  for( int v = 0; v < velocityT.length; v++ )
    for( int a0 = 0; a0 < anglesT.length; a0++ )
      for( int d0 = 0; d0 < distT.length; d0++ )
        for( int a1 = 0; a1 < anglesT.length; a1++ ){
              float[] input = { scaleVel(velocityT[v]), 
                                scaleAng(anglesT[a0]), 
                                scaleDst(distT[d0]), 
                                scaleAng(anglesT[a1])};
              //println(String.format("%.2f, %.2f, %.2f, %.2f, %.2f, %.2f",input[0],input[1],input[2],input[3],input[4],input[5]));
              float[] output;
              if( velocityT[v] < 1. )
                if( anglesT[a0] <= .10 ) output = new float[]{1,0,0,0};
                else                     output = new float[]{1,0,1,0};
              else
                if( anglesT[a0] <= .10 ) output = new float[]{0,0,0,0};
                else                     output = new float[]{0,0,1,0};              
              inputs .add( new Matrix( input));
              targets.add( new Matrix(output));
            }
  return new ArrayTrainer( inputs, targets, nn );
}

/*  Training the following classification
    | .00-.25 | .25-.50 | .50-.75 | .75-1.0 |
.00 |         |         |         |         |
.25 | Class 1 | Class 4 | Class 2 | Class 3 |
---------------------------------------------
.25 |         |         |         |         |
.50 | Class 5 | Class 3 | Class 1 | Class 4 |
---------------------------------------------
.50 |         |         |         |         |
.75 | Class 4 | Class 2 | Class 5 | Class 3 |
---------------------------------------------
.75 |         |         |         |         |
1.0 | Class 3 | Class 1 | Class 4 | Class 1 |
---------------------------------------------
*/

ArrayTrainer madeUpDataTrainer( NeuralNetwork nn ){
  ArrayList<Matrix> inputs = new ArrayList<Matrix>();  

  inputs.add( new Matrix( new float[]{.1,.1}));
  inputs.add( new Matrix( new float[]{.1,.2}));
  inputs.add( new Matrix( new float[]{.1,.3}));
  inputs.add( new Matrix( new float[]{.1,.4}));
  inputs.add( new Matrix( new float[]{.1,.6}));
  inputs.add( new Matrix( new float[]{.1,.7}));
  inputs.add( new Matrix( new float[]{.1,.8}));
  inputs.add( new Matrix( new float[]{.1,.9}));
  
  inputs.add( new Matrix( new float[]{.2,.1}));
  inputs.add( new Matrix( new float[]{.2,.2}));
  inputs.add( new Matrix( new float[]{.2,.3}));
  inputs.add( new Matrix( new float[]{.2,.4}));
  inputs.add( new Matrix( new float[]{.2,.6}));
  inputs.add( new Matrix( new float[]{.2,.7}));
  inputs.add( new Matrix( new float[]{.2,.8}));
  inputs.add( new Matrix( new float[]{.2,.9}));
  
  inputs.add( new Matrix( new float[]{.3,.1}));
  inputs.add( new Matrix( new float[]{.3,.2}));
  inputs.add( new Matrix( new float[]{.3,.3}));
  inputs.add( new Matrix( new float[]{.3,.4}));
  inputs.add( new Matrix( new float[]{.3,.6}));
  inputs.add( new Matrix( new float[]{.3,.7}));
  inputs.add( new Matrix( new float[]{.3,.8}));
  inputs.add( new Matrix( new float[]{.3,.9}));
  
  inputs.add( new Matrix( new float[]{.4,.1}));
  inputs.add( new Matrix( new float[]{.4,.2}));
  inputs.add( new Matrix( new float[]{.4,.3}));
  inputs.add( new Matrix( new float[]{.4,.4}));
  inputs.add( new Matrix( new float[]{.4,.6}));
  inputs.add( new Matrix( new float[]{.4,.7}));
  inputs.add( new Matrix( new float[]{.4,.8}));
  inputs.add( new Matrix( new float[]{.4,.9}));
    
  inputs.add( new Matrix( new float[]{.6,.1}));
  inputs.add( new Matrix( new float[]{.6,.2}));
  inputs.add( new Matrix( new float[]{.6,.3}));
  inputs.add( new Matrix( new float[]{.6,.4}));
  inputs.add( new Matrix( new float[]{.6,.6}));
  inputs.add( new Matrix( new float[]{.6,.7}));
  inputs.add( new Matrix( new float[]{.6,.8}));
  inputs.add( new Matrix( new float[]{.6,.9}));
  
  inputs.add( new Matrix( new float[]{.7,.1}));
  inputs.add( new Matrix( new float[]{.7,.2}));
  inputs.add( new Matrix( new float[]{.7,.3}));
  inputs.add( new Matrix( new float[]{.7,.4}));
  inputs.add( new Matrix( new float[]{.7,.6}));
  inputs.add( new Matrix( new float[]{.7,.7}));
  inputs.add( new Matrix( new float[]{.7,.8}));
  inputs.add( new Matrix( new float[]{.7,.9}));
  
  inputs.add( new Matrix( new float[]{.8,.1}));
  inputs.add( new Matrix( new float[]{.8,.2}));
  inputs.add( new Matrix( new float[]{.8,.3}));
  inputs.add( new Matrix( new float[]{.8,.4}));
  inputs.add( new Matrix( new float[]{.8,.6}));
  inputs.add( new Matrix( new float[]{.8,.7}));
  inputs.add( new Matrix( new float[]{.8,.8}));
  inputs.add( new Matrix( new float[]{.8,.9}));
  
  inputs.add( new Matrix( new float[]{.9,.1}));
  inputs.add( new Matrix( new float[]{.9,.2}));
  inputs.add( new Matrix( new float[]{.9,.3}));
  inputs.add( new Matrix( new float[]{.9,.4}));
  inputs.add( new Matrix( new float[]{.9,.6}));
  inputs.add( new Matrix( new float[]{.9,.7}));
  inputs.add( new Matrix( new float[]{.9,.8}));
  inputs.add( new Matrix( new float[]{.9,.9}));
  
  ArrayList<Matrix> targets = new ArrayList<Matrix>();  
  Matrix O1 = new Matrix( new float[]{1,0,0,0,0});
  Matrix O2 = new Matrix( new float[]{0,1,0,0,0});
  Matrix O3 = new Matrix( new float[]{0,0,1,0,0});
  Matrix O4 = new Matrix( new float[]{0,0,0,1,0});
  Matrix O5 = new Matrix( new float[]{0,0,0,0,1});
  for( int i = 0; i < 16; i++ ) targets.add(O1);
  for( int i = 0; i <  8; i++ ) targets.add(O2);
  for( int i = 0; i < 16; i++ ) targets.add(O3);
  for( int i = 0; i < 16; i++ ) targets.add(O4);
  for( int i = 0; i <  8; i++ ) targets.add(O5);
  return new ArrayTrainer( inputs, targets, nn );
}

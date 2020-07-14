class XYPlot{
  FloatList data;
  int       yMetric      = 0;
  int       iterations   = 0;
  float     initialError = 0;
  
  XYPlot(){
    data = new FloatList();  
  }
  
  void addPoint( float point ){
    if( initialError == 0 ) initialError = point;
/*    if( data.size() == width ){
      println(String.format("size: %d", data.size()));
      for( int i = data.size()-1; i >= 0; i-=2 )
        data.remove(i);
      println(String.format("iter: %d, size: %d", iterations, data.size()));
    }*/
    if( data.size() > width*2 ){
      for( int i = 0; i < width; i++ )
        data.remove(i);
    }        
    data.append(point);
    iterations++;
  }
  
  void render(){
    background(255);
    if( data.size() < 2 ) return;  
    beginShape();                                  //Draw trail
    stroke(color(255,0,0));
    strokeWeight(2);
    noFill();
    float maxVal = data.max();
    float x = 0, y = 0;
    for( int i = 0; i < data.size(); i++ ){
      x = map(          i , 0, data.size(),      0, width );
      y = map( data.get(i), 0,      maxVal, height,     0 );
      vertex(x,y);
    }
    endShape();
    
    fill(0);
    text( running ? "Running" : "Stopped"                                              ,          10, nextY(  true ));
    text(String.format("Iterations: %d"         , iterations)                          ,          10, nextY( false ));
    text(String.format("Initial error: %.2f"    , initialError)                        ,          10, nextY( false ));
    text(String.format("Points: %d"             , data.size())                         ,          10, nextY( false ));
    text(String.format("( %d, %.4f )"           , iterations-data.size(),data.get(0))  ,          10,           20  ); //top left
    text(String.format("( %d, 0 )"              , iterations-data.size())              ,          10,  height - 10  ); //bottom left
    text(String.format("Error: ( %d, %.4f )"    , iterations, data.get(data.size()-1)) , width - 150,  height /  2  ); //right middle
  }

  int nextY( boolean reset ){
    if (reset) yMetric = 60;
    yMetric += 20;
    return yMetric;
  }  
  
}

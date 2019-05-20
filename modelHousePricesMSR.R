library( tidyverse )
library( MicrosoftML )


bigData <- readr::read_csv( 'Melbourne_housing_FULL.csv' ) %>%
  mutate( Date = lubridate::dmy( Date ) )

minDate <- min( bigData$Date )

modelData <- bigData %>% 
  filter( Type == 'h'
          , Bedroom2 > 0
          , Bathroom > 0
          , Bedroom2 <= 5
          , Bathroom <=3
          , Car <=4
          , grepl( x=Regionname, pattern='Metro' ) ) %>%  
  mutate( id=factor( row_number() )
          , timeCount=as.numeric( Date-minDate )
          , timeFactor=cut( timeCount, c( 0, 365, 730, 1095 ) )
          , nearCity=Distance <= 10
          , roomFactor=factor( Rooms )
          , logPrice=log( Price ) ) %>%
  select( id
          , Suburb
          , Regionname
          , Date
          , timeCount
          , timeFactor
          , nearCity
          , roomFactor
          , Distance
          , Lattitude
          , Longtitude
          , Bedroom2
          , Bathroom
          , Car
          , Price
          , logPrice) %>%
  filter_all( all_vars( !is.na(.) ) )  %>%
  mutate_if( is.character, factor )

pricePlot <- ggplot( modelData, aes( x=Price ) ) + geom_histogram( bins=30 )
pricePlot + labs( x='Price', y='Count', title='Distribution of Melbourne house prices.' )
pricePlot + scale_x_log10( breaks=c( 1e5, 3e5, 1e6, 3e6 ) ) +
  labs( x='Price', y='Count', title='Distribution of Melbourne house prices'
        , subtitle='Logarithmic scale' )

set.seed( 0 )
perfData <- modelData %>% 
  sample_n( 25 )

aggMetrics <- NULL
aggPerf <- NULL



myRandomSeeds=c( 123, 42, 2019, 7, 13, 1337, 90210, 2358, 184594917, 99, 101 )

for( iterCount in 1:length( myRandomSeeds ) ){
  thisSeed <- myRandomSeeds[iterCount]

    set.seed( thisSeed )
  trainData <- modelData %>%
    sample_frac( size=0.75, replace=FALSE )
  
  testData <- modelData %>%
    anti_join( trainData, by='id' )
  
  set.seed( thisSeed )
  
  houseLM <- rxLinMod( logPrice ~ timeFactor + Regionname + nearCity + roomFactor
                       , data=trainData )
  
  set.seed( thisSeed )
  
  houseLM1 <- rxLinMod( logPrice ~ timeFactor + Suburb + nearCity + roomFactor
                        , data=trainData )
  
  set.seed( thisSeed )
  
  houseNN <- rxNeuralNet( logPrice ~ Distance + 
                            Suburb +
                            Bedroom2 + 
                            Bathroom + 
                            Car + 
                            timeCount
                          , data=trainData
                          , numHiddenNodes=5
                          , numIterations = 50
                          , type='regression' )
  predData <- testData %>%
    mutate( constPrice=log( 1e6 )
            , lmLogPrice=rxPredict( houseLM
                                  , data=. )[,1]
            , l2LogPrice = rxPredict( houseLM1, data=. )[,1]
            , nnLogPrice=rxPredict( houseNN
                                    , data = . )[,1]
    )

  predMetrics <- predData %>%
    summarise( constVar=var( logPrice - constPrice )
               , lmLogVar=var( (logPrice - lmLogPrice) )
               , l2LogVar = var( (logPrice-l2LogPrice) )
               , nnLogVar=var( (logPrice - nnLogPrice) )
               , iteration=iterCount
    )
  
  perfData <- perfData %>%
    mutate( constPrice=log( 1e6 )
            , lmLogPrice=rxPredict( houseLM
                                  , data=. )[,1]
            , l2LogPrice = rxPredict( houseLM1, data=. )[,1]
            , nnLogPrice=rxPredict( houseNN
                                    , data=. )[,1] 
            , iteration=iterCount )
  

  
  
  aggPerf <- perfData %>%
    select( id
            , constPrice
            , logPrice
            , lmLogPrice
            , l2LogPrice
            , nnLogPrice
            , iteration ) %>% 
    bind_rows( aggPerf )
  
  aggMetrics <- bind_rows( aggMetrics, predMetrics )
}

aggPlot <- ggplot( aggPerf, aes(x=id ) ) + 
  geom_boxplot( aes( y=constPrice, colour='Constant' ), alpha=0.6 ) +
  geom_boxplot( aes( y=nnLogPrice, colour='NN' ), alpha=0.6 ) + 
  geom_boxplot( aes( y=lmLogPrice, colour='LM' ), alpha=0.6 ) + 
  geom_point( aes( y=logPrice, colour='Actual'), shape=4, size=2 ) +
  theme( legend.position='bottom'
         , axis.text.x=element_text( angle=45 ) ) +
  labs( x='ID'
        , y='log( Price )'
        , title='Bias/Variance trade-off between LM and NN algorithms' ) +
  scale_colour_manual( values=c('Constant'='hotpink'
                                , 'LM'='goldenrod'
                                , 'NN'='darkviolet'
                                , 'Actual'='black' ) )

aggPlot1 <- ggplot( aggPerf, aes(x=id ) ) + 
  geom_boxplot( aes( y=constPrice, colour='Constant' ), alpha=0.6 ) +
  geom_boxplot( aes( y=nnLogPrice, colour='NN' ), alpha=0.6 ) + 
  geom_boxplot( aes( y=lmLogPrice, colour='LM' ), alpha=0.6 ) + 
  geom_boxplot( aes( y=l2LogPrice, colour='LM2' ), alpha=0.6 ) + 
  geom_point( aes( y=logPrice, colour='Actual'), shape=4, size=2 ) +
  theme( legend.position='bottom'
         , axis.text.x=element_text( angle=45 ) ) +
  labs( x='ID'
        , y='log( Price )'
        , title='Bias/Variance trade-off between LM and NN algorithms' ) +
  scale_colour_manual( values=c( 'Constant'='hotpink'
                                , 'LM'='goldenrod'
                                , 'LM2'='maroon'
                                , 'NN'='darkviolet'
                                , 'Actual'='black' ) )

aggPlot1

aggMetrics1 <- aggMetrics %>% 
  gather( 1:4, key=Model, value=Variance ) %>% 
  mutate( Model=case_when( grepl( x=Model, pattern='const' ) ~ 'Const'
                           , grepl( x=Model, pattern='lm' ) ~ 'LM'
                           , grepl( x=Model, pattern='l2' ) ~ 'LM2'
                           , TRUE ~ 'NN' ) )


metricPlot <- ggplot( aggMetrics1 %>% 
                        filter( Model=='Const' | Model == 'LM' | Model == 'NN' ) ) + 
  geom_boxplot( aes( x=Model, y=Variance, colour=Model ) ) + 
  scale_colour_manual( values=c('Const'='hotpink', 'LM'='goldenrod', 'NN'='darkviolet' ) ) +
  labs( x='Model'
        , y='Error Variance'
        , title='Cross-Validation of predictions from Constant, LM, and NN models') +
  theme( legend.position='bottom' )

metricPlot

metricPlot1 <- ggplot( aggMetrics1  ) + 
  geom_boxplot( aes( x=Model, y=Variance, colour=Model ) ) + 
  scale_colour_manual( values=c('Const'='hotpink'
                                , 'LM'='goldenrod'
                                , 'LM2' = 'maroon'
                                , 'NN'='darkviolet'
                                 ) ) +
  labs( x='Model'
        , y='Error Variance'
        , title='Cross-Validation of predictions from Constant, LM, and NN models') +
  theme( legend.position='bottom' )

aggPerfSummary <- aggPerf %>% 
  group_by( id ) %>% 
  summarise( lmVar=var( logPrice-lmLogPrice )
             , l2Var=var( logPrice-l2LogPrice )
             , nnVar=var( logPrice - nnLogPrice ) 
             )

predClassData <- predData %>% 
  select( id, logPrice, lmLogPrice, nnLogPrice ) %>% 
  mutate( constPrice=log( 1e6 )
          , constClass=as.integer( constPrice > logPrice )
          , lmClass=as.integer( lmLogPrice > logPrice )
          , nnClass=as.integer( nnLogPrice > logPrice )
          , ensembleClass=as.integer( lmClass + nnClass + constClass >= 2 ) 
          , contrarianClass=as.integer( 1L-ensembleClass )
          )

N_MC = 10000
initClassResult <- rep( NA_real_, N_MC )
mcData <- data.frame( constResult=initClassResult
                      , lmResult=initClassResult 
                      , nnResult=initClassResult
                      , ensembleResult=initClassResult 
                      , contrarianResult=initClassResult )

for( mcCount in 1:N_MC ){
  realClass <- as.integer( runif( nrow( predClassData ), min=0, max=1 ) <= 0.5 )
  thisResult <- predClassData %>%
    summarise( constResult=mean( as.integer( constClass == realClass ) )
               , lmResult=mean( as.integer( lmClass == realClass ) )
               , nnResult=mean( as.integer( nnClass == realClass ) )
               , ensembleResult=mean( as.integer( ensembleClass==realClass ) )
               , contrarianResult=mean( as.integer( ensembleClass != realClass ) )
    )
  mcData[mcCount,] <- thisResult
}

mcPlotData <- mcData %>% gather( key=Method, value=PCC )
mcPlot <- ggplot( mcPlotData, aes( x=PCC, fill=Method ) ) + 
  geom_histogram( binwidth = 0.005, alpha=0.5, position='identity' ) +
  labs( x='Probability of Correct Classification'
        , y='Number'
        , title='Performance of different algorithms'
        , subtitle='Based on Monte Carlo of different simulated performance sets' ) +
  theme( legend.position='bottom' )

mcPlot
mcPlot + facet_wrap( ~Method )

save.image( file='melbourneHousePrices.RData' )

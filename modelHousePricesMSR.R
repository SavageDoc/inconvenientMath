###################
## Demonstration code for 
## "Inconvenient Mathematics: 
## Why Data Scientists Can't Give Ugly Ducklings a Free Lunch"
# 
# Includes:
# * Example of bias-variance trade-off
# * Demonstration of the No Free Lunch Theorem
# 
## Utilises data from Kaggle regarding Melbourne House Prices:
# https://www.kaggle.com/anthonypino/melbourne-housing-market
##
## Uses Microsoft ML Server, which is free to use for non-production purposes.
## For more information, see, for example:
# https://docs.microsoft.com/en-us/machine-learning-server/install/machine-learning-server-windows-install
# 
# Or search for Microsoft ML Server with your favourite search engine.

## Written by Craig "Doc" Savage in support of his presentation 
# at the Citizens Data Science Meetup group in Melbourne, Australia, on 20 May, 2019.
# The slides have been released within the same GitHub repository. 
# Video of the presentation should be released "soon".
#
# Feel free to copy/fork/etc.

## Begin code ----
# Package load 
# tidyverse for general tools - data handling, plotting, etc.
library( tidyverse )
# MicrosoftML for machine learning functionality
# You may be able to replace this with other functions (e.g. lm, nnet, etc)
library( MicrosoftML )

## Data load ----
bigData <- readr::read_csv( 'Melbourne_housing_FULL.csv' ) %>%
  mutate( Date = lubridate::dmy( Date ) )

# Get the minimal date for reference
minDate <- min( bigData$Date )

# Data cleaning & coarsening
modelData <- bigData %>% 
  filter( Type == 'h' # Houses only
          , Bedroom2 > 0 # At least one bedroom
          , Bathroom > 0 # At least one bathroom
          , Bedroom2 <= 5 # Not "too big"
          , Bathroom <=3  # Not "too big"
          , Car <=4 # Not "too big"
          , grepl( x=Regionname, pattern='Metro' )  # Only metro - not rural
          ) %>%  
  mutate( id=factor( row_number() ) # Make an ID
          , timeCount=as.numeric( Date-minDate ) # Number of days since minDate
          , timeFactor=cut( timeCount, c( 0, 365, 730, 1095 ) ) # Cut into years
          , nearCity=Distance <= 10 # Boolean variable to be "near" the city
          , roomFactor=factor( Rooms ) # Consider rooms as a factor: Ordinal variable
          , logPrice=log( Price ) # Take the log of the price - model this as response
          ) %>%
  # Select only relevant variables
  select( id
          , Suburb
          , Regionname
          , Date
          , timeCount
          , timeFactor
          , nearCity
          , roomFactor
          , Distance
          , Bedroom2
          , Bathroom
          , Car
          , Price
          , logPrice) %>%
  # Remove missing values
  filter_all( all_vars( !is.na(.) ) )  %>%
  # Convert strings to factors
  mutate_if( is.character, factor )

## EDA ----
# Generate distribution of price
# Base plot
pricePlot <- ggplot( modelData, aes( x=Price ) ) + geom_histogram( bins=30 )
# Add labels
pricePlot + labs( x='Price', y='Count', title='Distribution of Melbourne house prices.' )
# Add lables & consider log scaling
pricePlot + scale_x_log10( breaks=c( 1e5, 3e5, 1e6, 3e6 ) ) +
  labs( x='Price', y='Count', title='Distribution of Melbourne house prices'
        , subtitle='Logarithmic scale' )

## Computations ----
# Initialise the random seed
set.seed( 0 )
# Get 25 random points to illustrate bias-variance trade-off
perfData <- modelData %>% 
  sample_n( 25 )

# Empty aggregated metrics and performance data frames
aggMetrics <- NULL
aggPerf <- NULL

# A few of my favourite seeds
myRandomSeeds=c( 123, 42, 2019, 7, 13, 1337, 90210, 2358, 184594917, 99, 101 )

# Simple cross-validation: For each seed
# 1) Split into training/test
# 2) Fit models - 2 linear models and neural net
# NB: The RevoScaleR functions (starting with rx) are proprietary, but open-source packages
# work in a similar fashion: with a formula and data, create model objects....
# 3) Get predictions on test data
# 4) Compute metric of predicted and actual on test data
for( iterCount in 1:length( myRandomSeeds ) ){
  # Abbreviate the seed
  thisSeed <- myRandomSeeds[iterCount]

    set.seed( thisSeed )
    # Split data into training/test
  trainData <- modelData %>%
    sample_frac( size=0.75, replace=FALSE )
  
  testData <- modelData %>%
    anti_join( trainData, by='id' )
  
  # Re-initialise the random seed
  set.seed( thisSeed )
  
  # Fit the first linear model (See slides/presentation for clarification)
  houseLM <- rxLinMod( logPrice ~ timeFactor + Regionname + nearCity + roomFactor
                       , data=trainData )
  
  set.seed( thisSeed )
  
  # Fit the second linear model (Note: Difference of formula - see slides/presentation)
  houseLM1 <- rxLinMod( logPrice ~ timeFactor + Suburb + nearCity + roomFactor
                        , data=trainData )
  
  set.seed( thisSeed )
  
  # Fit the neural network
  houseNN <- rxNeuralNet( logPrice ~ Distance + 
                            Suburb +
                            Bedroom2 + 
                            Bathroom + 
                            Car + 
                            timeCount
                          , data=trainData
                          # You may wish to play with the meta parameters
                          , numHiddenNodes=5 
                          , numIterations = 50
                          , type='regression' )
  
  # Generate predicitons
  predData <- testData %>%
    mutate( constPrice=log( 1e6 ) # Constant model - independent of training data
            # Other predictions via RevoScaleR -
            # Check the compatability if you've gone to alternate packages!
            , lmLogPrice=rxPredict( houseLM
                                  , data=. )[,1]
            , l2LogPrice = rxPredict( houseLM1, data=. )[,1]
            , nnLogPrice=rxPredict( houseNN
                                    , data = . )[,1]
    )

  # Calculate metrics
  predMetrics <- predData %>%
    # Using variance of the error....
    # This results in a "short, fat" results structure
    summarise( constVar=var( logPrice - constPrice )
               , lmLogVar=var( (logPrice - lmLogPrice) )
               , l2LogVar = var( (logPrice-l2LogPrice) )
               , nnLogVar=var( (logPrice - nnLogPrice) )
               # Store the iteration number
               , iteration=iterCount
    )
  
  # Update results for the randomly chosen sample of houses for the bias-variance demo
  perfData <- perfData %>%
    mutate( constPrice=log( 1e6 )
            , lmLogPrice=rxPredict( houseLM
                                  , data=. )[,1]
            , l2LogPrice = rxPredict( houseLM1, data=. )[,1]
            , nnLogPrice=rxPredict( houseNN
                                    , data=. )[,1] 
            , iteration=iterCount )
  
  # Aggregate
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

# Plot the results: Constant, LM, NN, Actual
# Only considers the random selection of houses
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

# Same plot as above, with LM2 added
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

# Aggregate the metrics - transform from short/fat to tidy
aggMetrics1 <- aggMetrics %>% 
  gather( 1:4, key=Model, value=Variance ) %>% 
  # Tighten up the lables
  mutate( Model=case_when( grepl( x=Model, pattern='const' ) ~ 'Const'
                           , grepl( x=Model, pattern='lm' ) ~ 'LM'
                           , grepl( x=Model, pattern='l2' ) ~ 'LM2'
                           , TRUE ~ 'NN' ) )

# Look at the variation of model performance with a box-plot
metricPlot <- ggplot( aggMetrics1 %>% 
                        filter( Model=='Const' | Model == 'LM' | Model == 'NN' ) ) + 
  geom_boxplot( aes( x=Model, y=Variance, colour=Model ) ) + 
  scale_colour_manual( values=c('Const'='hotpink', 'LM'='goldenrod', 'NN'='darkviolet' ) ) +
  labs( x='Model'
        , y='Error Variance'
        , title='Cross-Validation of predictions from Constant, LM, and NN models') +
  theme( legend.position='bottom' )

metricPlot

# As above, but without filtering out LM2
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

# Look at the summary of the variance
aggPerfSummary <- aggPerf %>% 
  group_by( id ) %>% 
  summarise( lmVar=var( logPrice-lmLogPrice )
             , l2Var=var( logPrice-l2LogPrice )
             , nnVar=var( logPrice - nnLogPrice ) 
             )

# Classification example (No Free Lunch Theorem) ----
# Generate classifications based on model predictions of "true" worth
predClassData <- predData %>% 
  select( id, logPrice, lmLogPrice, nnLogPrice ) %>% 
  mutate( constPrice=log( 1e6 )
          # Classes are binary - here flagged 0/1
          , constClass=as.integer( constPrice > logPrice )
          , lmClass=as.integer( lmLogPrice > logPrice )
          , nnClass=as.integer( nnLogPrice > logPrice )
          , ensembleClass=as.integer( lmClass + nnClass + constClass >= 2 ) 
          , contrarianClass=as.integer( 1L-ensembleClass )
          )

# Begin Monte Carlo simulation
N_MC = 10000
# Initialise results vector
initClassResult <- rep( NA_real_, N_MC )
# ... and associated data frame
mcData <- data.frame( constResult=initClassResult
                      , lmResult=initClassResult 
                      , nnResult=initClassResult
                      , ensembleResult=initClassResult 
                      , contrarianResult=initClassResult )

# Run the simulations
for( mcCount in 1:N_MC ){
  # Uniformly generate "true" classes
  # NB: This is the contestable assumption of the No Free Lunch Theorem
  # I agree that uniform is not the best assumption - but what is?
  realClass <- as.integer( runif( nrow( predClassData ), min=0, max=1 ) <= 0.5 )
  # Summarise results
  thisResult <- predClassData %>%
    summarise( constResult=mean( as.integer( constClass == realClass ) )
               , lmResult=mean( as.integer( lmClass == realClass ) )
               , nnResult=mean( as.integer( nnClass == realClass ) )
               , ensembleResult=mean( as.integer( ensembleClass==realClass ) )
               , contrarianResult=mean( as.integer( ensembleClass != realClass ) )
    )
  # Populate results data frame
  mcData[mcCount,] <- thisResult
}

# Plot results
# Tidy data set
mcPlotData <- mcData %>% gather( key=Method, value=PCC )
# Build plot
mcPlot <- ggplot( mcPlotData, aes( x=PCC, fill=Method ) ) + 
  geom_histogram( binwidth = 0.005, alpha=0.5, position='identity' ) +
  labs( x='Probability of Correct Classification'
        , y='Number'
        , title='Performance of different algorithms'
        , subtitle='Based on Monte Carlo of different simulated performance sets' ) +
  theme( legend.position='bottom' )

mcPlot
# Separate the plots by Method rather than overlap
mcPlot + facet_wrap( ~Method )

# Save the results for next time....
save.image( file='melbourneHousePrices.RData' )

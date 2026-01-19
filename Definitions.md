**Metrics Definitions**

Scale:
- **Orders**: number of orders delivered by couriers
    The metric defines the scale and importance of the market to the business
    
    Use case: Where the greates performance lever is?
    *Lagging* indicator

- **Growth Index**: weighted average of order growth over the last 12 months
    Recent months are weighted more heavily (linear weighting: 1, 2, 3... oldest to newest) to capture trend changes while considering longer-term patterns.
    
    Use case: How is the market trending? Is growth accelerating or decelerating? Can we predict future market size and resource needs?
    *Leading* indicator

Supply health metrics:
- *Delivery CPO*: cost per order in euros associated with the delivery of the order
    Assumptions:
    - for simplicity we assume this is a direct variable cost only (promo, courier pay & incentives). This does not include indirect variable costs (insurance, refunds, chargebacks, support costs)

    Use case: Is the unit economy of delivery reasonable to scale the market or require optimisation?
    Problem with the metric: the dataset does not provide the revenue side for unit economy calculations, so we can't reasonably make assumption about economic efficiency of any market. Even the higher Delivery CPO may not be a problem if market fieatures higher Average Check and user tolerance towards delivery fees
 
    *Lagging* indicator

- *OPH*: average number of orders completed by 1 courier per operating hour (orders per hour)
    The metric represent courier utilisation. Forcasting demand, Scheduling, Dispatch, and Courier-facing products will all have this as the core optimisation KPI. 

    Use case: How effectively do we utilise the available couriers?
    High OPH may be a native feature of a dence urban market but may also be a sign of overutilisation and burnout of the couriers, so should be monitored rogether with courier churn
    Low OPH would be a feature of less dence markets (large travel distance), but also oversupplied by couriers. It has a critical impact on courier earnings and Full Unit economy efficiency

    *Leading* indicator

- *Closing*: % of time that the delivery network is closed (i.e. customers cannot place orders) because there aren't enough couriers to fulfill the demand
    Use case: Where do we experience critical undersupply issues? 
    It means the market either grows faster than the courier base or courier supply deteriorates for any reason. May be a sign of failure of the Forcasting demand and Scheduling algorithms, or more broad courier acquisition.

    *Leading* indicator

User experience metrics
- *PtoD*: average time in minutes from customer placing an order to courier arriving at the customer's door (placed to delivered)
    Use case: How much user needs to wait their order?
    This includes merchant preparation time, order pickup time, delivery time. 
    Deterioration of this metric may represent courier supply issues, failure of the algoritm predicting the order pickup time from merchant. Severe issues with PtoD likely to impact on user retention 

    *Leading* indicator

- *UDO*: % of orders that customers claim to not have been delivered and for which a refund is provided
    Use case: Critical metric relates to customer trust. 
    If order is not delivered, users are much more likely to churn in the future. 
    High UDO may be an indication of undersupply, operational issues with courier onboarding/tracking, or fraud detection gaps

    *Lagging* indicator
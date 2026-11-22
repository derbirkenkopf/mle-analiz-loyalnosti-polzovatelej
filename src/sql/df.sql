select
  user_id,
  device_type_canonical,
  order_id,
  created_dt_msk order_dt,
  created_ts_msk order_ts,
  currency_code,
  revenue,
  tickets_count,
  (
    created_dt_msk::date - lag(created_dt_msk::date)
    over
    (
      partition by user_id order by created_dt_msk
    )
  ) as days_since_prev,
  p.event_id,
  event_name_code event_name,
  event_type_main,
  service_name,
  region_name,
  city_name
from afisha.purchases p
join afisha.events e using(event_id)
join afisha.city c using(city_id)
join afisha.regions r using(region_id)
where device_type_canonical in ('mobile', 'desktop')
and e.event_type_main != 'фильм'
order by user_id, created_dt_msk;

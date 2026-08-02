-- Dimension tables for lookup and enrichment. These tables are used by
-- the materialized views to add human-readable dimension values to event data.
CREATE TABLE inmobi_cat.advertisers
(
    advertiser_id  String,
    vertical       LowCardinality(String),
    campaign_type  Enum8('CPM' = 1, 'CPC' = 2, 'CPI' = 3)
)
ENGINE = SharedMergeTree
ORDER BY advertiser_id;

CREATE TABLE inmobi_cat.apps
(
    app_id          String,
    category        LowCardinality(String),
    publisher_tier  Enum8('tier_1' = 1, 'tier_2' = 2, 'tier_3' = 3)
)
ENGINE = SharedMergeTree
ORDER BY app_id;

CREATE TABLE inmobi_cat.geo_device
(
    geo_device_id  String,
    region         Enum8('APAC' = 1, 'NAM' = 2, 'EU' = 3, 'LATAM' = 4, 'MEA' = 5),
    country        FixedString(2),
    device_model   LowCardinality(String),
    os_version     LowCardinality(String)
)
ENGINE = SharedMergeTree
ORDER BY geo_device_id;

-- Raw event table containing fact-level ad impressions, fills, clicks, and revenue.
CREATE TABLE inmobi_cat.ad_events
(
    event_time     DateTime64(3),
    app_id         String,
    geo_device_id  String,
    advertiser_id  String,
    ad_format      LowCardinality(String),
    is_filled      UInt8,
    is_impression  UInt8,
    is_click       UInt8,
    revenue        Float64,

    INDEX idx_advertiser_id advertiser_id TYPE bloom_filter GRANULARITY 4,
    INDEX idx_geo_device_id geo_device_id TYPE bloom_filter GRANULARITY 4
)
ENGINE = SharedMergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (event_time, app_id, ad_format);

-- Hourly aggregated view used by the hourly anomaly grain.
CREATE TABLE inmobi_cat.ad_events_hourly_agg
(
    hour             DateTime,
    dimension_name   LowCardinality(String),
    dimension_value  LowCardinality(String),
    requests         UInt64,
    fills            UInt64,
    impressions      UInt64,
    clicks           UInt64,
    revenue          Float64
)
ENGINE = SummingMergeTree((requests, fills, impressions, clicks, revenue))
ORDER BY (dimension_name, dimension_value, hour);

-- Daily aggregated view used by the daily anomaly grain.
CREATE TABLE inmobi_cat.ad_events_daily_agg
(
    date             Date,
    dimension_name   LowCardinality(String),
    dimension_value  LowCardinality(String),
    requests         UInt64,
    fills            UInt64,
    impressions      UInt64,
    clicks           UInt64,
    revenue          Float64
)
ENGINE = SummingMergeTree((requests, fills, impressions, clicks, revenue))
ORDER BY (dimension_name, dimension_value, date);

-- Dictionaries for fast lookup during aggregation.
CREATE DICTIONARY inmobi_cat.advertisers_dict
(
    advertiser_id String,
    vertical String,
    campaign_type String
)
PRIMARY KEY advertiser_id
SOURCE(CLICKHOUSE(
    QUERY 'SELECT advertiser_id, toString(vertical) AS vertical, toString(campaign_type) AS campaign_type FROM inmobi_cat.advertisers'
    USER 'default' PASSWORD ''
))
LAYOUT(HASHED())
LIFETIME(MIN 300 MAX 600);

CREATE DICTIONARY inmobi_cat.apps_dict
(
    app_id String,
    category String,
    publisher_tier String
)
PRIMARY KEY app_id
SOURCE(CLICKHOUSE(
    QUERY 'SELECT app_id, toString(category) AS category, toString(publisher_tier) AS publisher_tier FROM inmobi_cat.apps'
    USER 'default' PASSWORD ''
))
LAYOUT(HASHED())
LIFETIME(MIN 300 MAX 600);

CREATE DICTIONARY inmobi_cat.geo_device_dict
(
    geo_device_id String,
    country String,
    region String,
    device_model String,
    os_version String
)
PRIMARY KEY geo_device_id
SOURCE(CLICKHOUSE(
    QUERY 'SELECT geo_device_id, toString(country) AS country, toString(region) AS region, toString(device_model) AS device_model, toString(os_version) AS os_version FROM inmobi_cat.geo_device'
    USER 'default' PASSWORD ''
))
LAYOUT(HASHED())
LIFETIME(MIN 300 MAX 600);

-- Materialized view that builds daily aggregates across all dimensions.
CREATE MATERIALIZED VIEW inmobi_cat.mv_ad_events_daily_agg
TO inmobi_cat.ad_events_daily_agg
AS
SELECT
    toDate(event_time) AS date,
    dim.1 AS dimension_name,
    dim.2 AS dimension_value,
    count()             AS requests,
    sum(is_filled)       AS fills,
    sum(is_impression)   AS impressions,
    sum(is_click)        AS clicks,
    sum(revenue)         AS revenue
FROM inmobi_cat.ad_events
ARRAY JOIN
    [
        ('__total__',      '__total__'),
        ('ad_format',      toString(ad_format)),
        ('app_id',         app_id),
        ('category',       dictGet('inmobi_cat.apps_dict', 'category', app_id)),
        ('publisher_tier', dictGet('inmobi_cat.apps_dict', 'publisher_tier', app_id)),
        ('vertical',       dictGetOrDefault('inmobi_cat.advertisers_dict', 'vertical', advertiser_id, 'unknown')),
        ('campaign_type',  dictGetOrDefault('inmobi_cat.advertisers_dict', 'campaign_type', advertiser_id, 'unknown')),
        ('country',        dictGet('inmobi_cat.geo_device_dict', 'country', geo_device_id)),
        ('region',         dictGet('inmobi_cat.geo_device_dict', 'region', geo_device_id)),
        ('device_model',   dictGet('inmobi_cat.geo_device_dict', 'device_model', geo_device_id)),
        ('os_version',     dictGet('inmobi_cat.geo_device_dict', 'os_version', geo_device_id))
    ] AS dim
GROUP BY date, dimension_name, dimension_value;

-- Materialized view that builds hourly aggregates across all dimensions.
CREATE MATERIALIZED VIEW inmobi_cat.mv_ad_events_hourly_agg
TO inmobi_cat.ad_events_hourly_agg
AS
SELECT
    toStartOfHour(event_time) AS hour,
    dim.1 AS dimension_name,
    dim.2 AS dimension_value,
    count()             AS requests,
    sum(is_filled)       AS fills,
    sum(is_impression)   AS impressions,
    sum(is_click)        AS clicks,
    sum(revenue)         AS revenue
FROM inmobi_cat.ad_events
ARRAY JOIN
    [
        ('__total__',      '__total__'),
        ('ad_format',      toString(ad_format)),
        ('app_id',         app_id),
        ('category',       dictGet('inmobi_cat.apps_dict', 'category', app_id)),
        ('publisher_tier', dictGet('inmobi_cat.apps_dict', 'publisher_tier', app_id)),
        ('vertical',       dictGetOrDefault('inmobi_cat.advertisers_dict', 'vertical', advertiser_id, 'unknown')),
        ('campaign_type',  dictGetOrDefault('inmobi_cat.advertisers_dict', 'campaign_type', advertiser_id, 'unknown')),
        ('country',        dictGet('inmobi_cat.geo_device_dict', 'country', geo_device_id)),
        ('region',         dictGet('inmobi_cat.geo_device_dict', 'region', geo_device_id)),
        ('device_model',   dictGet('inmobi_cat.geo_device_dict', 'device_model', geo_device_id)),
        ('os_version',     dictGet('inmobi_cat.geo_device_dict', 'os_version', geo_device_id))
    ] AS dim
GROUP BY hour, dimension_name, dimension_value; 



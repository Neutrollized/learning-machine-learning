# README
I assume the site will be taken down or change at some point so I'm including a PDF'd version of the original page.  The prize is laughably shitty (offering $1500 in credits for a recommendation engine/algorithm/model that can make them millions); honestly the time/cost to train such a model to do what they're asking for would likely cost that much already.

That being said, the [dataset](../../data/booking-com_challenge_2020.zip) contains a lot of useful real world data that can be used for learning and thus I'm including it here.

## Dataset
The training dataset consists of over a million of anonymized hotel reservations, based on real data, with the following features:
- **user_id** - User ID
- **check-in** - Reservation check-in date
- **checkout** - Reservation check-out date
- **affiliate_id** - An anonymized ID of affiliate channels where the booker came from (e.g. direct, some third party referrals, paid search engine, etc.)
- **device_class** - desktop/mobile
- **booker_country** - Country from which the reservation was made (anonymized)
- **hotel_country** - Country of the hotel (anonymized)
- **city_id** - city_id of the hotel’s city (anonymized)
- **utrip_id** - Unique identification of user’s trip (a group of multi-destinations bookings within the same trip)
Each reservation is a part of a customer’s trip (identified by utrip_id) which includes at least 4 consecutive reservations. The check-out date of a reservation is the check-in date of the following reservation in their trip.

The evaluation dataset is constructed similarly, however the city_id of the final reservation of each trip is concealed and requires a prediction.

## Evaluation criteria
The goal of the challenge is to predict (and recommend) the final city (city_id) of each trip (utrip_id). We will evaluate the quality of the predictions based on the top four recommended cities for each trip by using Precision@4 metric (4 representing the four suggestion slots at Booking.com website). When the true city is one of the top 4 suggestions (regardless of the order), it is considered correct.

### Submission guidelines
The test set will be released to registered e-mails on January 14st, 2021. The teams are expected to submit their top four city predictions per each trip on the test set until January 28th 2021. The submission should be completed on easychair website  (https://easychair.org/conferences/?conf=bookingwebtour21). in a csv file named submission.csv with the following columns;
- **utrip_id** - 1000031_1
- **city_id_1** - 8655
- **city_id_2** - 8652
- **city_id_3** - 4323
- **city_id_4** - 4332

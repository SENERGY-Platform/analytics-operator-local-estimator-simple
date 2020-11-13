#  Copyright 2020 InfAI (CC SES)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import typing
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import *
import numpy as np
from senergy_local_analytics import Input, Output, App


class LinReg:

    def __init__(self):
        self.Sxy, self.Sx, self.n, self.alpha, self.beta, self.x_avg, self.y_avg, self.x_0 = 0, 0, 0, 0, 0, 0, 0, 0

    def lr(self, new_x, new_y) -> None:
        """
        x_avg: average of previous x, if no previous sample, set to 0
        y_avg: average of previous y, if no previous sample, set to 0
        Sxy: covariance of previous x and y, if no previous sample, set to 0
        Sx: variance of previous x, if no previous sample, set to 0
        n: number of previous samples
        new_x: new incoming 1-D numpy array x
        new_y: new incoming 1-D numpy array y
        """
        if self.n == 0:
            self.x_0 = new_x[0]
            new_x[0] = 0

        new_n = self.n + len(new_x)

        new_x_avg = (self.x_avg * self.n + np.sum(new_x)) / new_n
        new_y_avg = (self.y_avg * self.n + np.sum(new_y)) / new_n

        if self.n > 0:
            x_star = (self.x_avg * np.sqrt(self.n) + new_x_avg * np.sqrt(new_n)) / (np.sqrt(self.n) + np.sqrt(new_n))
            y_star = (self.y_avg * np.sqrt(self.n) + new_y_avg * np.sqrt(new_n)) / (np.sqrt(self.n) + np.sqrt(new_n))
        elif self.n == 0:
            x_star = new_x_avg
            y_star = new_y_avg
        else:
            raise ValueError

        new_Sx = self.Sx + np.sum((new_x - x_star) ** 2)
        new_Sxy = self.Sxy + np.sum((new_x - x_star).reshape(-1) * (new_y - y_star).reshape(-1))

        self.beta = 0
        if new_Sx > 0:
            self.beta = new_Sxy / new_Sx
        self.alpha = new_y_avg - self.beta * new_x_avg
        self.Sxy = new_Sxy
        self.Sx = new_Sx
        self.n = new_n
        self.x_avg = new_x_avg
        self.y_avg = new_y_avg

    def train(self, current_timestamp, value):
        x, y = np.array([current_timestamp - self.x_0]), np.array([value])
        self.lr(x, y)

    def predict(self, target_timestamp):
        return (target_timestamp-self.x_0)*self.beta


lr = LinReg()


def process(inputs: typing.List[Input]):
    today = datetime.utcnow().date()
    eoy = datetime(today.year, 12, 31)
    eom = datetime(today.year, today.month, 1) + relativedelta(months=1)
    eod = datetime(today.year, today.month, today.day) + timedelta(days=1)
    value = 0
    ts = ""
    message_id = ""
    for inp in inputs:
        if inp.name == "value" and inp.current_value is not None:
            value = inp.current_value
        if inp.name == "timestamp" and inp.current_value is not None:
            ts = inp.current_value
        if inp.name == "message_id" and inp.current_value is not None:
            message_id = inp.current_value
    timestamp = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc).timestamp()
    lr.train(timestamp, value)
    pred_day = lr.predict(eod.replace(tzinfo=timezone.utc).timestamp())
    pred_month = lr.predict(eom.replace(tzinfo=timezone.utc).timestamp())
    pred_year = lr.predict(eoy.replace(tzinfo=timezone.utc).timestamp())
    return Output(True, {"pred_day": pred_day, "pred_day_timestamp": str(eod),
                         "pred_month": pred_month, "pred_month_timestamp": str(eom),
                         "pred_year": pred_year, "pred_year_timestamp": str(eoy),
                         "message_id": message_id,
                         "timestamp": ts
                         })


if __name__ == '__main__':
    app = App()

    input1 = Input("value")
    input2 = Input("timestamp")
    input3 = Input("message_id")

    app.config([input1, input2, input3])
    print("start operator", flush=True)
    app.process_message(process)
    app.main()









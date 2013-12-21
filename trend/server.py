from flask import Flask, render_template, request
from trend import TrendVis
import json
app = Flask(__name__)
app.debug = True

trend = TrendVis()

@app.route('/trend/')
def trendvis():
	query = request.args.get('q') or ''
	print 'rendering trends for', query
	return render_template("trend.htm")

@app.route('/data/trend/')
def render_trend():
    q = request.args.get('q') or ''
    start = request.args.get('start') or 0
    end = request.args.get('end') or 10000
    print 'rendering terms for', q, 'between', start, "and", end
    return json.dumps(trend.query_terms(q, start_time=int(start), end_time=int(end)))

# @app.route('/data/trend/')
# def topic_trends():
#     query = request.args.get('q') or ''
#     # threshold = request.query.threshold or ''
#     # print 'rendering trends for', q, threshold, 'on', data
#     return trend.query_topic_trends(q, float(threshold))


if __name__ == "__main__":
	app.run(host='0.0.0.0',port=5002)

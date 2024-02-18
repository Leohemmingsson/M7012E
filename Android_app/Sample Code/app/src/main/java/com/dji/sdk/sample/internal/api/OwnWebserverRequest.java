package com.dji.sdk.sample.internal.api;

import android.os.AsyncTask;
import android.util.Log;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;

public class OwnWebserverRequest extends AsyncTask<String, Void, Map<String, String>> {
    private static final String TAG = "HttpGetRequest";
    private final OnRequestCompleteListener callback;

    public OwnWebserverRequest(OnRequestCompleteListener callback) {
        this.callback = callback;
    }

    @Override
    protected Map<String, String> doInBackground(String... params) {
        Log.i(TAG, "Do in background started");
        String urlString = params[0];
        Map<String, String> resultMap = new HashMap<>();
        HttpURLConnection urlConnection = null;

        try {
            URL url = new URL(urlString);
            urlConnection = (HttpURLConnection) url.openConnection();

            InputStream inputStream = urlConnection.getInputStream();
            InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
            BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
            StringBuilder stringBuilder = new StringBuilder();
            String line;

            while ((line = bufferedReader.readLine()) != null) {
                stringBuilder.append(line).append("\n");
            }

            String result = stringBuilder.toString();
            resultMap.put("result", result); // You can put any key-value pairs you need
        } catch (IOException e) {
            Log.e(TAG, "Error making GET request", e);
        } finally {
            if (urlConnection != null) {
                urlConnection.disconnect();
            }
        }

        return resultMap;
    }

    @Override
    protected void onPostExecute(Map<String, String> resultMap) {
        if (callback != null) {
            callback.onRequestComplete(resultMap);
        }
    }

    public interface OnRequestCompleteListener {
        void onRequestComplete(Map<String, String> resultMap);
    }
}


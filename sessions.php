<?php

header('Content-Type: application/json; charset=utf-8');

// Configuration
define('TENSORBOARD_BASE', 'https://stats.ailive.co');
define('DIRECTORY_PATH', '/var/mlwork/public/sessions/zero/walking/tensorboard');

$response = new stdClass();
$response->sessions = [];

// Ensure the directory exists
if (!is_dir(DIRECTORY_PATH)) {
    http_response_code(400);
    die(json_encode(['error' => 'The specified directory does not exist.']));
}

// Status Codes
const STATUS_CREATED = 0;
const STATUS_IN_QUEUE = 1;
const STATUS_LIVE = 2;
const STATUS_PAUSED = 3;
const STATUS_FINISHED = 4;
const STATUS_TERMINATED = 5;

$meanEp = getMeanEpLengths();
$items = scandir(DIRECTORY_PATH);
$latestSessionId = 0;

// Process tensorboard directories
foreach ($items as $item) {
    if ($item === '.' || $item === '..') {
        continue;
    }

    $itemPath = DIRECTORY_PATH . DIRECTORY_SEPARATOR . $item;
    if (is_dir($itemPath)) {
        $sessionId = extractSessionId($item);

        if (!isset($meanEp->runToSeries->{$item})) {
            continue; // Skip if no mean episode length data is available
        }

        $epMeanData = $meanEp->runToSeries->{$item};
        $lastValue = end($epMeanData)->value ?? 0;

        $session = new stdClass();
        $session->id = "zero_walking_$sessionId";
        $session->date = filemtime($itemPath);
        $session->status = STATUS_FINISHED;
        $session->score = calculateScore($lastValue);
        $session->replay_url = "https://ml.ailive.co/sessions/zero/walking/obs/$sessionId.json";
        $session->stats_url = TENSORBOARD_BASE . "/?darkMode=false&runFilter=_{$sessionId}_#timeseries";
        $response->sessions[] = $session;

        if ($sessionId > $latestSessionId) {
            $latestSessionId = $sessionId;
        }
    }
}

// Mark the latest session as live
foreach ($response->sessions as $session) {
    if ($latestSessionId == extractSessionId($session->id)) {
        $session->status = STATUS_LIVE;
    }
}

// Sort sessions by date in descending order
usort($response->sessions, fn($a, $b) => $b->date <=> $a->date);

echo json_encode($response);

/**
 * Extract the session ID from a directory name.
 *
 * @param string $directoryName
 * @return int|null
 */
function extractSessionId(string $directoryName): ?int
{
    $parts = explode("_", $directoryName);
    return isset($parts[2]) ? (int) $parts[2] : null;
}

/**
 * Calculate a score based on the given value.
 *
 * @param float $value
 * @return float
 */
function calculateScore(float $value): float
{
    $maxValue = 1000;
    if ($value < 0) {
        return 0; // Handle negative values
    }
    $value = min($value, $maxValue); // Cap the value at maxValue
    return floor(10000 * sqrt($value / $maxValue)) / 100;
}

/**
 * Fetch the mean episode lengths from TensorBoard.
 *
 * @return object|null
 */
function getMeanEpLengths(): ?object
{
    $url = TENSORBOARD_BASE . '/experiment/defaultExperimentId/data/plugin/timeseries/timeSeries';

    $postData = <<<EOD
------WebKitFormBoundaryPlR0UgggiDNp9QtZ
Content-Disposition: form-data; name="requests"

[{"plugin":"scalars","tag":"rollout/ep_len_mean"}]
------WebKitFormBoundaryPlR0UgggiDNp9QtZ--
EOD;

    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $postData);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'Accept: application/json, text/plain, */*',
        'Content-Type: multipart/form-data; boundary=----WebKitFormBoundaryPlR0UgggiDNp9QtZ',
    ]);
    curl_setopt($ch, CURLOPT_SSL_VERIFYHOST, false);
    curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, false);

    $response = curl_exec($ch);
    if (curl_errno($ch)) {
        error_log('cURL Error: ' . curl_error($ch));
        return null;
    }

    curl_close($ch);
    return json_decode($response);
}